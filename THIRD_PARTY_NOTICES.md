# Third-party notices

Project copyright: Copyright (c) 2026 Shuyang Zeng. Copyright is retained with
limited non-production evaluation permission under the [Portfolio Evaluation License](LICENSE).
This license does not replace
any applicable third-party copyright, license, or notice obligations, including
the licenses of separately installed software.

The simulator uses Python's standard library at runtime. This repository and its
wheel do not bundle the Python interpreter or standard-library implementation.
Python's license and notices are provided by its distribution; see the
[Python license documentation](https://docs.python.org/3/license.html).

## Build and test dependencies

The development requirements install the following packages. They are tools used
to build or test the simulator, not vendored implementations in its source or wheel.
The installed distributions retain their own license files and metadata.

| Package | Use | Official project |
|---|---|---|
| setuptools | Build backend and packaging | [setuptools](https://github.com/pypa/setuptools) |
| pytest | Test runner | [pytest](https://github.com/pytest-dev/pytest) |
| iniconfig | Test runner configuration dependency | [iniconfig](https://github.com/pytest-dev/iniconfig) |
| packaging | Package/version handling dependency | [packaging](https://github.com/pypa/packaging) |
| pluggy | Test runner plugin dependency | [pluggy](https://github.com/pytest-dev/pluggy) |
| Pygments | Test output highlighting dependency | [Pygments](https://github.com/pygments/pygments) |

Version pins are in [requirements-dev.txt](requirements-dev.txt). Building or
testing with these dependencies does not make their implementations part of this
project's authorship. Anyone redistributing an environment or bundled dependencies
must also retain the licenses and notices applicable to those distributions.

## Distribution boundary

This version does not include or require ABIDES, its matching modules, or an ABIDES
adapter. It does not bundle NumPy, pandas, or other simulation backends. No upstream
project's affiliation or endorsement is claimed.

Any future addition of copied or adapted third-party material must preserve its
applicable copyright, license, notices, and source attribution. The project's
Portfolio Evaluation License cannot be used to remove those requirements.
