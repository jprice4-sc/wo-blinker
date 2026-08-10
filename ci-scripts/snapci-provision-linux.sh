#!/bin/bash
# Script used to provision the Snap CI Linux image used for running the CI pipelines.

install_base_requirements () {
    # preparatory
    sudo apt-get update

    # Download
    sudo apt-get install -y curl wget git git-lfs
    # Build
    sudo apt-get install -y make libssl-dev pkg-config libhdf5-dev
    # Deploy
    sudo apt-get install -y zip patchelf python3 python3-pip python3.12-venv
    # Lint
    sudo apt-get install -y lsb-release software-properties-common
    # Machamp
    sudo apt-get install -y apt-utils apt-transport-https build-essential ca-certificates gcc gnupg rsync ssh
}

install_npm_cspell () {
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash
    # Source the nvm env variables to use it straight in the current shell
    source "$HOME/.nvm/nvm.sh"
    nvm install --latest-npm --lts
    npm --version
    npm install -g cspell
    cspell --version
}

install_rustup () {
    # NOTE: Needed for cargo even if rust isn't used in repo
    curl --proto '=https' --tlsv1.3 https://sh.rustup.rs -sSf | sh -s -- -y
    # Source the rust env variables to use it straight in the current shell
    source "$HOME/.cargo/env"
    # Check installed ok
    rustup --version

}

install_taplo() {
    # NOTE: Depends on rustup install for cargo
    cargo install taplo-cli
    taplo --version
}

install_uv_and_python_tools () {
    # Install the latest uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the uv env variables to use it straight in the current shell
    source $HOME/.local/bin/env

    uv --version

    # Install twine
    uv tool install twine
    uv tool install --index https://registry.snapchat.com/python/virtual/ --native-tls wo-doctopus
    twine --version
    doctopus --version
}

install_gcloud_cli () {
    # Google Cloud CLI, required for gsutil https://cloud.google.com/storage/docs/gsutil_install#deb
    sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
    # Import the Google Cloud public key
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg
    # Add the gcloud CLI distribution URI as a package source
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
    # Update and install the gcloud CLI
    sudo apt-get update && sudo apt-get install -y google-cloud-cli
}

main () {
    set -x -euo pipefail

    echo "Provisioning the image..."

    install_base_requirements
    install_rustup
    install_taplo
    install_uv_and_python_tools
    install_gcloud_cli
    install_npm_cspell

    # configure git
    git config --global user.email "snapci@snapchat.com"
    git config --global user.name "snapci"

    echo "Cleaning up..."
    sudo apt-get clean
}

if [[ "$(uname)" != "Linux" ]]; then
    echo "This script is intended for Linux!"
    exit 1
fi

main "$@"
