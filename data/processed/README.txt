电力负荷预测数据集说明
==================================================

数据来源: Load_ALL.xlsx
处理日期: 2025-10-22 15:53:12

数据集结构:
- Area1_Preprocessed.csv: Area1完整数据集
- Area2_Preprocessed.csv: Area2完整数据集
- Area1_train.csv: Area1训练集 (70%)
- Area1_val.csv: Area1验证集 (15%)
- Area1_test.csv: Area1测试集 (15%)
- Area2_train.csv: Area2训练集 (70%)
- Area2_val.csv: Area2验证集 (15%)
- Area2_test.csv: Area2测试集 (15%)

字段说明:
- datetime: 时间戳 (YYYY-MM-DD HH:MM:SS)
- load: 电力负荷 (MW)
- temp_avg: 平均温度 (℃)
- humidity: 相对湿度 (%)
- rain: 降雨量 (mm)
- area: 区域标识 (Area1/Area2)

数据特点:
- 时间频率: 每15分钟一个数据点
- 每天数据点数: 96个
- 数据时间范围: 2012-01-01 00:00:00 到 2015-01-10 23:45:00
- Area1数据量: 105984 条记录
- Area2数据量: 105984 条记录
