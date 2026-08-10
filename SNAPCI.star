# SNAPCI.star defines configuration for SnapCI
# For information on SnapCI see
# https://sci-docsite.mesh.sc-corp.net/
# https://sci-docsite.mesh.sc-corp.net/docs/reference/starlark/

###################
# VM image lookup #
###################

image.vm(
    name = "ci_linux_image",
    base = "snap-ubuntu-2404",
    provision_with = "ci-scripts/snapci-provision-linux.sh",
)

linux_exec_requirements = {
    "os": "linux",
    "arch": "x86_64", 
    "vm_image": "#<REPLACE_WITH_HASH_AFTER_IMAGE_CREATED>//ci_linux_image", # image.vm() as defined at specific commit
    "ttl": "100",
}

login_bash = ["bash", "-l", "-c"]

############################
# Pipeline config - CHECKS #
############################

run(
    name = "check-spelling",
    description = "Spell check using cspell",
    steps = [
        process(*(login_bash + ["cspell lint ./**/*.{c,cpp,h,md,py,rs,rst,sh,star,toml,ts,yaml}"])),
    ],
    exec_requirements = linux_exec_requirements,
)

run(
    name = "check-lint",
    description = "Lint check using ruff",
    steps = [
        process(*(login_bash + ["uvx ruff --version && uvx ruff check"])),
    ],
    exec_requirements = linux_exec_requirements,
)

run(
    name = "check-format",
    description = "Format check using ruff",
    steps = [
        process(*(login_bash + ["uvx ruff --version && uvx ruff format --diff"])),
    ],
    exec_requirements = linux_exec_requirements,
)

run(
    name = "check-types",
    description = "Type check using mypy",
    steps = [
        process(*(login_bash + ["uv run mypy --version && uv run mypy"])),
    ],
    exec_requirements = linux_exec_requirements,
)

run(
    name = "check-tests",
    description = "Run pytest on the repo",
    steps = [
        # process(*(login_bash + ["echo '3.13' && uv run --python=3.13 pytest"])),
        process(*(login_bash + ["echo '3.12' && uv run --python=3.12 pytest"])),
    ],
    exec_requirements = linux_exec_requirements,
)

run(
    name = "check-docs",
    description = "Check that pydoctor compiles docs for all documented python packages in the repo.",
    steps = [
        process(*(login_bash + ["source ci-scripts/ci_functions.sh && check_docs_of_all_pyprojects"])),
    ],
    exec_requirements = linux_exec_requirements,
)

run(
    name = "check-toml-format",
    description = "Check that pyproject.toml files are formatted correctly using taplo",
    steps = [
        process(*(login_bash + ["taplo format --check --diff --config taplo.toml **/pyproject.toml"])),
    ],
)

# Use the system python since pip-audit requires the stock pip
# TODO: change with uv audit once available, https://github.com/astral-sh/uv/issues/9189
security_check = [
    "echo 'PIP-AUDIT'",
    "uv export --all-extras --frozen --no-editable --no-emit-project --no-hashes --format requirements-txt --output-file requirements-uv.txt",
    "uvx --python-preference=only-system pip-audit -r requirements-uv.txt",
]

run(
    name = "check-dev-environment",
    description = "Check all the tools are functioning and updated",
    steps = [
        process(*(login_bash + [" && ".join(security_check)])),
        process(*(login_bash + ["echo 'LOCK' && uv lock --locked"])),
        process(*(login_bash + ["echo 'BUILD' && uv build --wheel"])),
    ],
    exec_requirements = linux_exec_requirements,
)

on_pr(
    execs = [
        "check-spelling",
        "check-lint",
        "check-format",
        "check-types",
        "check-tests",
        "check-docs",
        "check-dev-environment",
    ],
    continue_on_failure = True,
)

on_comment(
    name = "checks",
    body = "/check",
    execs = [
        "check-spelling",
        "check-lint",
        "check-format",
        "check-types",
        "check-tests",
        "check-docs",
        "check-dev-environment",
    ],
    continue_on_failure = True,
)

################################################
# Pipeline config - PUBLISH ARTIFACTS and DOCS #
################################################

run(
    #NOTE: ONLY WORKS FOR on_tag
    name = "publish-wheels",
    description = "Publish wheels using uv",  # TODO: change to uv publish
    steps = [
        process(*(login_bash + ["source ci-scripts/ci_functions.sh && check_tag_and_project_versions_match && build_and_publish_wheel_of_tag"])),
    ],
    exec_requirements = linux_exec_requirements,
)

run(
    #NOTE: ONLY WORKS FOR on_tag
    name = "publish-docs",
    description = "Publish docs using doctopus",
    steps = [
        process(*(login_bash + ["source ci-scripts/ci_functions.sh && check_tag_and_project_versions_match && build_and_publish_docs_of_tag"])),
    ],
    exec_requirements = linux_exec_requirements,
)

# Regex explained:
# Regex is designed to match tags such as the following
# where 1.2.3 is a placeholder for a python canonical version specifier
#
# release=v1.2.3
# release=mypackage/v1.2.3
#
# The part of the regex which looks like this ([a-zA-Z_-]+/)?
# allows the optional mypackage/ prefix which is needed for multi package repos
# 
# The rest of the regex after the v is 
# pulled from Python's canonical form of version specifiers: 
# https://packaging.python.org/en/latest/specifications/version-specifiers/#appendix-parsing-version-strings-with-regular-expressions
on_tag(
    name = "publish_artifacts_and_docs",
    tag = match.regex(r'^release=([a-zA-Z0-9_-]+/)?v([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$'),
    execs = [
        "publish-wheels",
        "publish-docs",
    ],
    continue_on_failure = False,
)
