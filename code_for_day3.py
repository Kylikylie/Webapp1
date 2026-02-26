最终输出结构
行结构
每行 = 国家 × 年份 × 性别
预期行数：约 204国家 × 2年份 × 3性别 = 1,224行
列结构
列序
列名
说明
数据来源
A
Year
年份(2015/2018)
-
B
Sex
性别(Male/Female/Both)
-
C
Location
国家全名
GBD
D
Label
国家缩写(ISO3)
GDD/World Bank
E
X
GDP per capita (current US$)
World Bank
F
Y
TB发病率
GBD
G
upper
TB发病率95%CI上限
GBD
H
lower
TB发病率95%CI下限
GBD
I
Population
人口数
World Bank
J-ET
v01-v57
47种膳食摄入×3列(median/upper/lower)=141列
GDD
EU-FF
混杂因素
人均卫生支出、HIV、糖尿病、吸烟率各×3列=12列
World Bank/GBD
 
 
 
 
 
总列数：约 9列基础信息 + 141列膳食摄入 + 12列混杂因素 = 162列
数据处理步骤
Step 1: 读取GBD TB数据（主表）
读取TB发病率数据，筛选需要的列：· location_name → Location· sex_name → Sex· year → Year· val → Y· upper, lower → CI
Step 2: 读取World Bank数据
· GDP数据：筛选2015和2018年，提取REF_AREA(Label)、OBS_VALUE(X)· 人口数据：按性别分别读取· 人均卫生支出：筛选对应年份
Step 3: 读取GDD数据（47种膳食摄入）
筛选条件：· age = 999（全年龄）· female in [0, 1, 999]（男/女/合计）· urban = 999（城乡合计）· edu = 999（教育程度合计）· year in [2015, 2018]提取列：iso3, year, female, median, upperci_95, lowerci_95
Step 4: 读取GBD混杂因素数据
· HIV和糖尿病患病率· 吸烟SEV
Step 5: 建立国家名称映射
GBD、World Bank、GDD三者的国家名称/代码需要匹配：· GBD: location_name（国家全名）· World Bank: REF_AREA（ISO3代码）· GDD: iso3（ISO3代码）
Step 6: 合并所有数据
以GBD TB数据为主表，依次左连接其他数据源
Step 7: 输出Excel
输出文件：整理后数据.xlsx
关键数据字段对照
GBD数据字段
· location_name: 国家名称
· sex_name: 性别 (Male/Female/Both)
· year: 年份 (2015/2018)
· val: 点估计值
· upper: 95%CI上限
· lower: 95%CI下限
World Bank数据字段
· REF_AREA: 国家缩写 (ISO3)
· REF_AREA_LABEL: 国家全名
· TIME_PERIOD: 年份
· OBS_VALUE: 观测值
GDD数据字段
· iso3: 国家缩写 (ISO3)
· year: 年份
· female: 性别编码 (0=男, 1=女, 999=合计)
· age: 年龄编码 (999=全年龄)
· urban: 城乡编码 (0=农村, 1=城市, 999=合计)
· edu: 教育程度编码 (1/2/3/999=合计)
· median: 点估计值
· upperci_95: 95%CI上限
· lowerci_95: 95%CI下限
