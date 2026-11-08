# DPCU-PolSfP 参考实现（中文说明）

**重要：这不是作者当初真实实验的原始代码。** 本仓库根据论文中的公式与文字重新编写，用于展示实现思路与提供可运行的实验框架；没有原始网络源码、MWIR图像、扫描网格、训练权重或相机SDK。不能据此声称已复现论文的 6.58° / 7.08° / 1.91%D / 1.25%D 结果。

## 本仓库包括

- Stokes四角度计算、Fresnel热发射-环境反射有效模型、DoLP查表反演、双候选法向；
- 局部/上下文双分支、预测熵门控、ST Gumbel-Softmax及约束损失；
- 法向角误差、oracle误差、选择误差、深度RMSE、Chamfer和LSMR积分；
- 训练、测试、物理参数敏感性、门控消融、五种子稳定性、决策反转分析；
- 数据文件格式、环境安装、自动化测试与GitHub Actions。

## 最简运行

```bash
pip install -e '.[test]'
pytest -q
dpcu-polsfp demo-data --out data/demo
dpcu-polsfp train --train data/demo/train.csv --val data/demo/val.csv --epochs 2 --width 8 --out weights/toy.pt
dpcu-polsfp evaluate --manifest data/demo/test.csv --checkpoint weights/toy.pt
```

这个demo只用于排查程序是否能执行。因为真实数据、真实模型、仪器驱动均未提供，任何生成的toy指标都不能写进论文，也不能说这份代码就是论文原始代码。**先拿真实源码和权重逐模块比对，验证论文核心结果，再正式发布声明。**

详细接口和论文匹配情况请阅读英文 `README.md` 和 `docs/AUTHOR_ACTION_REQUIRED.md`。
