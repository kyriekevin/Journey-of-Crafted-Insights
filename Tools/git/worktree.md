---
tags:
  - git
  - worktree
date: 2025-01-18T16:00:00
---
# Worktree

## Why?

场景：
1. 正在main分支上跑训练任务，切换到feat分支开发，训练任务就会中断
2. 项目大，频繁切换索引，成本高
3. 切换分支的环境变量和配置可能不一致
4. 切换同事代码分支，可能和自己的代码冲突
5. ByteDance内部对仓库个数有限制，不方便新建仓库

前几个问题可能可以`git clone`多个repo来解决，但是最后一个往往只能通过在一个repo中切换分支来解决。
1. 多个repo状态不好同步
2. git history/log是重复的，.git占用空间大
3. repo对应分支可能实际是多个repo，每个repo的环境依赖配置不同

## Introduction

`git worktree --help` 查看worktree作用：仅需维护一个repo，又可以同时在多个branch上工作，互不影响

![image.png](https://raw.githubusercontent.com/kyriekevin/img_auto/main/Obsidian/202501181705515.png)

> 默认情况下，`git init` 或者 `git clone` 初始化的repo只有一个worktree，叫做main worktree

### Commands

`git worktree add`
`alias: gwta='git worktree add'`

![image.png](https://raw.githubusercontent.com/kyriekevin/img_auto/main/Obsidian/202501181725927.png)

> [!NOTE] 如果repo为空直接创建分支会遇到如下报错 `fatal: not a valid object name: 'HEAD'`，因为没有初始化提交

`git worktree list`
`alias: gwtls='git worktree list'`

![image.png](https://raw.githubusercontent.com/kyriekevin/img_auto/main/Obsidian/202501181726468.png)

> [!NOTE] worktree 的工作做完要及时删除，否则会占用大量空间

`git worktree remove`

![image.png](https://raw.githubusercontent.com/kyriekevin/img_auto/main/Obsidian/202501181734097.png)

> [!NOTE] 删除worktree不会删除分支，只是删除了工作目录

## How to use?

```bash
md repo
cd repo

git clone --bare git@github.com:xxx.git .bare
# or
git init --bare .bare
```

这样就以bare形式clone项目代码，并将内容clone到.bare目录下

```bash
echo "gitdir: ./.bare" > .git
vim .bare/config
```

```plaintext
[remote "origin"]
    url = git@github.com:xxx.git
    fetch =+refs/heads/*:refs/remotes/origin/*
```

需要在项目目录下创建`.git`文件，文件内容是以`gitdir`的形式指向`.bare`文件夹，然后编辑`.bare/config`修改`[remote "origin"]`内容，确保我们创建worktree切换分支，可以显示正确分支名

```bash
git worktree add main
```

接着就可以创建worktree，首先为main分支创建一个worktree

![image.png](https://raw.githubusercontent.com/kyriekevin/img_auto/main/Obsidian/202501181757675.png)

因为 `.bare` 和 `.git` 是隐藏文件，所以只会有不同文件夹对应的工作区。这样在不同工作区分支上的操作就互不影响了
