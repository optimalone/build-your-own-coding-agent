# Reader-code publishing workflow

This repository is the public reader-code destination for
`/Users/owen/personal/build-a-coding-agent-manuscript/`, which is the source of
truth for the chapter snapshots.

This repository is an explicit exception to the general feature-pipeline PR
workflow: **do not create pull requests here**.

When the reviewed manuscript code is ready for publication:

1. Sync the chapter and appendix code from the manuscript repository using its
   existing sync script. Check source/destination paths when using a worktree.
2. Verify the synced files and run the chapter and appendix tests.
3. Commit and push directly to this repository's default branch as the GitHub
   account **`optimalone`**. Verify the push uses that identity; do not assume the
   active GitHub account is correct or publish as `owenthereal`.

Preserve this repository's separately maintained README, LICENSE, .gitignore,
.env.example, community implementations, and agent instructions during sync.
The manuscript repository retains its own review/PR workflow.
