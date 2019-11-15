# 如何参与到QPanda

我们很欢迎您能参与到QPanda的项目里来。不过在这之前，我们需要制定一些需要遵守的规则，以更好的发展我们的QPanda。

## 推送流程

为了保证项目的质量，所有的提交，包括项目成员的提交，都需要通过审查才能合并到项目中，所以我们需要使用Github pull requests。
[GitHub Help](https://help.github.com/articles/about-pull-requests/) 包含如何创建pull request的帮助信息.

在创建pull request之前，需要先fork QPanda [repo](https://github.com/OriginQ/QPanda-2)，然后使用这个fork中分支向QPanda的官方仓库创建 pull requests。在创建pull request时应选择推送到QPanda官方仓库的develop分支。

fork 和 pull request的基本流程如下：

1. Fork QPanda的仓库 [repo page](https://github.com/quantumlib/Cirq)。并把你的克隆仓库下载到你的本机。

1. 从master分支创建一个新的分支
    ```shell
    git checkout master -b new_branch_name
    ```
     ```new_branch_name``` 是你的新分支的名称.
1. 把你的修改提交到你自己的分支.
1. 如果你的克隆仓库与QPanda的官方仓库不同步，你需要先更新你的克隆仓库的master分支，然后再把master分支合并到你自己的分支:
    ```shell
    # Update your local master.
    git fetch upstream
    git checkout master
    git merge upstream/master
    # Merge local master into your branch.
    git checkout new_branch_name
    git merge master
    ```
    在合并的过程中，你可能要修改一些合并冲突.
1. 把你修改的推送到你克隆的仓库
    ```shell
    git push origin new_branch_name
    ```
1. 经过以上的操作，你就可以把你工作pull request给QPanda的官方仓库了，在pull request需要选择官方仓库的develop分支。我们将提供pull request的模板，请根据模板提供相关信息. 
1. 审查人员将对您的代码，并可能要求更改，您可以在本地执行这些操作，然后再次执行上述过程.

# 测试

在pull request您的代码之前，请针对您修改的代码编写单元测试，并需要通过现有的测试。QPanda的测试基于googletest，[googletest使用文档](https://github.com/google/googletest/blob/master/googletest/docs/primer.md)包含如何使用googletest编写单元测试。

在编写单元测试之前，您需要先注意一些规范：
1. 单元测试文件的命名必须是你要测试的功能+.test.cpp;
2. 使用Test(),第一个参数必须是你要测试的功能，第二个是要测试的内容；
3. 单元测试的代码规范需要遵守QPanda的代码规范。
