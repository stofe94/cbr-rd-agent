# Contributing

Suggestions for the further development of CBR-RD-Agent are welcome.

## Issues

Open an issue for bugs, questions and ideas, for example adapting the CBR similarity
heuristics and quality gates to further competitions. For a larger change, please open
an issue before writing code, so that the approach can be agreed first.

## Pull requests

- Open pull requests against `main`.
- Every change is reviewed by the maintainer, and only changes approved by the
  maintainer are accepted.
- Accepted changes are taken over into the next release rather than merged directly,
  so the pull request is closed with a reference to the release that contains it.
- Contributions are licensed under the MIT License, like the project (see
  [LICENSE](LICENSE)).

Before opening a pull request, please:

- keep the style of the surrounding code
- keep documentation and comments in English
- check that the agent starts and completes one loop, e.g. with `DS_LOOP_N=1` in
  `config.env` and `bash run_linux.sh` (or `exe_docker_linux.sh`)
- describe what the change does and how it was tested
