# Contributors Guide

Interested in improving xarray-subset-grid?
Have a few minutes to tackle an issue? Or improve the documentation?

## Introduction

First off, thank you for considering contributing to xarray-subset-grid.
Following these guidelines helps to communicate that you respect the time of the developers managing and developing these open source projects.
In return, they should reciprocate that respect in addressing your issue,
assessing changes, and helping you finalize your pull requests.
So, please take a few minutes to read through this guide.

## What Can I Do?

* Tackle any issues you wish!
* Contribute code you already have. It does not need to be perfect! We will help you clean
  things up, test it, etc.
* Make a tutorial or example of how to do something.
* Improve documentation of a feature you found troublesome.
* File a new issue if you run into problems!

## Ground Rules

The goal is to maintain a diverse community that's pleasant for everyone. Please
be considerate and respectful of others by following our
[code of conduct](https://github.com/ioos/.github/blob/main/CODE_OF_CONDUCT.md).

* Each pull request should consist of a logical collection of changes.
  You can include multiple bug fixes in a single pull request,
  but they should be related.
  For unrelated changes, please submit multiple pull requests.
* Do not commit changes to files that are irrelevant to your feature or bug fix
  (eg: .gitignore).
* Be willing to accept criticism and work on improving your code; we don't want
  to break other users' code, so care must be taken not to introduce bugs.
* Be aware that the pull request review process is not immediate, and is
  generally proportional to the size of the pull request.
* PRs with new features, or bugfixes, should not fix existing lints. The more changes in a PR, the harder it will be for the reviewer. We can always send a separate PR with the lint fix later.
* Avoid making the project "look like your project." We tend to make ourselves at home when writing contributions, but we should avoid renaming things just for renaming sake, moving files around to "organize" the project, or make aesthetic changes without asking the upstream devs first. Such changes, while sometimes desirable, don't bring much value to a PR but do bring extra burden to the reviewers.
* Do not tackle more than one issue or one feature per PR. While tempting, the smaller the PR, the easier will be to get it merged, reducing merge conflicts, without holding other PRs. Many small ones are always better than a single massive PR.
* Always state your PR objective clearly in the title "New Feature X", "Bug Fix Y", or "Fixing doc typo".
* Always add a "longer, but not too long, description of your PR in the body of the first comment.
* Always write meaningful, short, concise, commit messages. The kind of message your future self will congratulate you.
* When copying code from other places, like Stack-Overflow (SO), always add the link. That is not only a way to credit the original author of the code, but also add a way to look for the source when debugging. Also, it is not uncommon for a SO answer to be updated with better code.
* When using AI, also add a comment that you did! And, if possible, add the questions/chat report in a gist and link to it in the PR. While AI can be of great aid, it can also write obfuscated code that is hard to debug if you don't know it was machine generated.
* Be sure to search the existing issues and PRs before submitting yours. Sometimes these changes are already there and all we need is an extra person on the hill to make the final push. Be that second person and not the one that starts a new hill climb by yourself.
Always read the development notes when they exist, many questions are already answered there.

## Reporting a bug

The easiest way to get involved is to report issues you encounter when using IOOS Software or by
requesting something you think is missing.

* Head over to the project issues page.
* Search to see if your issue already exists or has even been solved previously.
* If you indeed have a new issue or request, click the "New Issue" button.
* Fill in as much of the issue template as is relevant. Please be as specific as possible.
  Include the version of the code you were using, as well as what operating system you
  are running. If possible, include complete, minimal example code that reproduces the problem.

## Setting up your development environment

We recommend using the [conda](https://conda.io/docs/) or [pixi](https://prefix.dev/) package managers for your Python environments.
Please take some time to go over the [README](https://github.com/asascience-open/xarray-subset-grid/blob/main/README.md).

Install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
(link with instructions) on your system if not already available
(check with ``git --version`` at the command line.)

Login to your [GitHub](https://github.com) account and make a fork of the
repository by clicking the "Fork" button.
Clone your fork of the repository (in terminal on Mac/Linux or git shell/GUI on Windows)
to the location you'd like to keep it.
We are partial to creating a ``git_repos`` or ``projects`` directory in our home folder.

```sh
git clone https://github.com/asascience-open/xarray-subset-grid.git
```

## Pull Requests

The changes to the code source (and documentation)
should be made via GitHub pull requests against ``main``,
even for those with administration rights.
While it's tempting to make changes directly to ``main`` and push them up,
it is better to make a pull request so that others can give feedback.
If nothing else,
this gives a chance for the automated tests to run on the PR.
This can eliminate "brown paper bag" moments with buggy commits on the main branch.

Push to your fork and submit a pull request.

## What happens after the pull request

You've made your changes, documented them, added some tests, and submitted a pull request.
What now?

### Code Review

At this point you're waiting on us. You should expect to hear at least a comment within a
couple of days. We may suggest some changes or improvements or alternatives.

Some things that will increase the chance that your pull request is accepted quickly:

* Write tests.
* Fix any failed lints shown by pre-commit-ci.
* Write a [good commit message](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html).

Pull requests will automatically have tests run by CIs.
This includes running both the unit tests as well as the code linters.
The test suite, documentation, style, and more will be checked on various versions of Python with current and legacy packages.
CIs will run testing on Linux, and Mac, and Windows.

### Merging

Once we're all happy with the pull request, it's time for it to get merged in. Only the
maintainers can merge pull requests and you should never merge a pull request you have commits
on as it circumvents the code review. If this is your first or second pull request, we'll
likely help by rebasing and cleaning up the commit history for you. As your development skills
increase, we'll help you learn how to do this.

## Further Reading

There are a ton of great resources out there on contributing to open source and on the
importance of writing tested and maintainable software.

* [How to Contribute to Open Source Guide](https://opensource.guide/how-to-contribute/)
* [Zen of Scientific Software Maintenance](https://jrleeman.github.io/ScientificSoftwareMaintenance/)
