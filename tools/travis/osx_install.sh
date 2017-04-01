#!/usr/bin/env bash

set -e -x

shell_session_update() { :; }

brew update
brew install swig

eval "$(pyenv init -)"

case "${PYTOK}" in
    py34)
        export TICK_PYVER="3.4.5"
        ;;
    py35)
        export TICK_PYVER="3.5.2"
        ;;
esac

env PYTHON_CONFIGURE_OPTS="--enable-framework" pyenv install -s $TICK_PYVER
pyenv global $TICK_PYVER

python -m pip install --quiet numpy pandas cpplint sphinx pillow bokeh -U pip
python -m pip install -r requirements.txt

pyenv rehash

if [ ! -d googletest ] || [ ! -f googletest/CMakeLists.txt ]
    then
        git clone https://github.com/google/googletest.git
        (cd googletest && mkdir -p build && cd build && cmake .. && make -s)
fi

export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

(cd googletest && cd build && sudo make -s install)
