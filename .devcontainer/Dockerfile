FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# This base image ships with a broken Yarn apt source (dl.yarnpkg.com)
# whose GPG key isn't trusted — apt-get update fails hard on it
# ("repository is not signed"), which breaks every devcontainer Feature
# that needs to apt-get install anything (confirmed: this is what broke
# docker-outside-of-docker's own install script). Not something a Python
# project needs anyway — removed rather than fixing its key.
RUN rm -f /etc/apt/sources.list.d/yarn.list

