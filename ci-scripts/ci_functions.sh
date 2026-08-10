#!/bin/bash
# Functions used in the CI executions defined in the SNAPCI.star file

_package_name_from_git_tag_or_root_pyproject() {
    local tag=$1
    
    # i.e. tag is release=v1.2.3
    if [[ $tag != */* ]]; then
        taplo get project.name --file-path pyproject.toml

    else
      # i.e. tag is release=mypackage/v1.2.3
      local tmp=${tag#release=}
      echo "${tmp%/*}"
    fi
}

_find_pyproject_of_package() {
    local package_name=$1
    
    for pyproject_file in $(find ./ -name pyproject.toml); do
        
        proj_name=$(taplo get project.name --file-path $pyproject_file)
        if [[ $proj_name == $package_name ]]; then
            echo $pyproject_file
            return 0
        fi
    done
}

_version_from_git_tag() {
    local tag=$1

    if [[ $tag != */* ]]; then
        # Expect tag to be of the form
        # "release=v1.2.3"
        # Remove the "release=v" prefix
        local version=${tag#release=v}
    else
        # Expect the tag to be of the form
        # "release=mypackage/v1.2.3"
        # Remove the "release=mypackage/v" prefix
        local version=${tag#release=*\/v}
    fi

    echo $version
}

check_docs_of_all_pyprojects() {
    set -euo pipefail
    # Iterate through all potential sub pyprojects (works for just one) and run pydoctor
    for pyproject_file in $(find ./ -name pyproject.toml); do
        if taplo get tool.pydoctor.html-output --file-path $pyproject_file &>/dev/null; then
            project_dir=$(dirname $pyproject_file)
            (
                set -euo pipefail
                cd $project_dir
                uvx pydoctor
            )
        fi
    done
}

build_and_publish_docs_of_tag() {
    set -euo pipefail
    local tag=${1:-$CI_TAG}

    local package_name=$(_package_name_from_git_tag_or_root_pyproject $tag)
    local pyproject_file=$(_find_pyproject_of_package $package_name)
    project_dir=$(dirname $pyproject_file)
    local version=$(taplo get project.version --file-path $pyproject_file)

    ( # In the potential subproject directory
        set -euo pipefail # needed as in new subshell
        cd $project_dir
        local docs_dir=$(taplo get tool.pydoctor.html-output --file-path pyproject.toml)
        rm -rf $docs_dir
        uvx pydoctor
        doctopus upload --name $package_name --version $version --dir $docs_dir
    )

    # Note: html-output is a relative to the pyproject file directory
    # so need to make correct relative to the CWD
}

check_tag_and_project_versions_match() {
    set -euo pipefail
    local tag=${1:-$CI_TAG}

    local package_name=$(_package_name_from_git_tag_or_root_pyproject $tag)
    local version=$(_version_from_git_tag $tag)
    local pyproject_file=$(_find_pyproject_of_package $package_name)
    local pyproject_version=$(taplo get project.version --file-path $pyproject_file)
    if [[ $version != $pyproject_version ]]; then
        echo "Tag version ($version) does not match pyproject version ($pyproject_version) in $pyproject_file"
        exit 1
    fi
}

build_and_publish_wheel_of_tag() {
    set -euo pipefail
    local tag=${1:-$CI_TAG}

    local package_name=$(_package_name_from_git_tag_or_root_pyproject $tag)

    rm -rf snap-ci-owned-dist 
    uv build --package=$package_name --wheel --out-dir snap-ci-owned-dist
    twine upload --verbose --repository snapchat --config-file .pypirc --cert /var/lib/snapci/workspace/executor_proxy/proxy.crt snap-ci-owned-dist/*.whl
}