# 2021华数杯全国大学生数学建模竞赛 
## 题目C：电动汽车目标客户销售策略研究

汽车产业是国民经济的重要支柱产业，而新能源汽车产业是战略性新兴产业。大力发展以电动汽车为代表的新能源汽车是解决能源环境问题的有效途径，市场前景广阔。但是，电动汽车毕竟是一个新兴的事物，与传统汽车相比，消费者在一些领域，如电池问题，还是存在着一些疑虑，其市场销售需要科学决策。
某汽车公司最新推出了三款品牌电动汽车，包括合资品牌（用1表示）、自主品牌（用2表示）和新势力品牌（用3表示）。为研究消费者对电动汽车的购买意愿，制定相应的销售策略，销售部门邀请了1964位目标客户对三款品牌电动汽车进行体验。具体体验数据有电池技术性能（电池耐用和充电方便）满意度得分（满分100分，下同）a1、舒适性（环保与空间座椅）整体表现满意度得分a2、经济性（耗能与保值率）整体满意度得分a3、安全性表现（刹车和行车视野）整体满意度得分a4、动力性表现（爬坡和加速）整体满意度得分a5、驾驶操控性表现（转弯和高速的稳定性）整体满意度得分a6、外观内饰整体表现满意度得分a7、配置与质量品质整体满意度得分a8等。另外还有目标客户体验者个人特征的信息，详情见附录1和2。

请你研究数据，查阅相关文献，运用数学建模的知识回答下列问题：

1. 请做数据清洗工作，指出异常值和缺失数据以及处理方法。对数据做描述性统计分析，包括目标客户对于不同品牌汽车满意度的比较分析。

2. 决定目标客户是否购买电动车的影响因素有很多，有电动汽车本身的因素，也有目标客户个人特征的因素。在这次目标客户体验活动中，有部分目标客户购买了体验的电动汽车（购买了用1表示，没有购买用0表示）。结合这些信息，请研究哪些因素可能会对不同品牌电动汽车的销售有影响？

3. 结合前面的研究成果，请你建立不同品牌电动汽车的客户挖掘模型，并评价模型的优良性。运用模型判断附件3中15名目标客户购买电动车的可能性。

4. 销售部门认为，满意度是目标客户汽车体验的一种感觉，只要营销者加大服务力度，在短的时间内提高a1-a8五个百分点的满意度是有可能的，但服务难度与提高的满意度百分点是成正比的，即提高体验满意度5%的服务难度是提高体验满意度1%服务难度的5倍。基于这种思路和前面的研究成果，请你在附件3每个品牌中各挑选1名没有购买电动汽车的目标客户，实施销售策略。

5. 根据前面的研究结论，请你给销售部门提出不超过500字的销售策略建议。

## 解决方法和思路

+ 主要解决方法(岭回归模型预处理数据、KDE核密度估计验证回归模型)
+ 主成分分析方法(PCA模型)
+ 基于属性对齐的门控神经网络模型

## 运行方法

在命令行终端创建虚拟环境
```bash
python -m venv mathmodelenv
source mathmodelenv/bin/activate
```
安装对应的包文件
```bash
pip install -r requirements.txt
```
## 主要问题结果的运行
1. 问题一的求解
```bash
python run.py --do-first \
--raw-file "附录1 目标客户体验数据.xlsx" \
--alpha 1.2 \
--beta 0.1 \
--batch-size 50 \
--raw-path "./Data" \
----sect-type normal \
--processed-path "./processed" \
---pics-path "./pictures" \
--log-path "./log"
```
2. 问题二的求解
```bash
python run.py --do-second \
--processed-path "./processed" \
--log-path "./log"
```
3. 问题三的求解
```bash
python run.py --do-third \
--test-file "附录3 待判定的数据.xlsx" \
--processed-path "./processed" \
--vec-type "embedding" \
--train-batch-size 400 \
--test-batch-size 400 \
--train-times 3000 \
--percentage 0.80 \
--log-path "./log"
```
