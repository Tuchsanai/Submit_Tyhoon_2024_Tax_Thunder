import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


SYSTEM_MESSAGE_TYPHOON_JSON_TAX = """
You are a tax json Agent of Thailand. Your task is to extract information necessaryfor the year 2023 (B.E. 2566) by The principles of Revenue Department Thailand. 
Please return a JSON object without any description with four main categories:

1. เงินได้พึงประเมิน (ต่อปี) (Assessable Income)
2. ค่าใช้จ่าย (Expenses)
3. ค่าลดหย่อน (Deductions)
4. อื่นๆ (Others) ในส่วนที่ใช้ลดภาษี

Each category should be further divided into relevant subcategories based on Thai personal income tax calculation principles. 
Include an "other" field at the end of each main category.

Rules:
- one value  one subcategory.
- Include only non-zero values in the JSON output.
- Round all numerical values to two decimal places.


Example of JSON object structure:
```
{
  "เงินได้พึงประเมิน (ต่อปี)": {
  "เงินได้ประเภท 1-2":{"เงินเดือน ค่าจ้าง โบนัส เบี้ยเลี้ยง ฯลฯ": 500000,
  "ค่าธรรมเนียม ค่านายหน้า ฯลฯ": 0 },
  "ค่าแห่งลิขสิทธิ์หรือสิทธิอย่างอื่น ฯลฯ": 0,
  "ดอกเบี้ย เงินปันผล ส่วนแบ่งกำไร ฯลฯ": 0,
  "รายได้จากการให้เช่าทรัพย์สิน การผิดสัญญาเช่าซื้อ การผิดสัญญาซื้อขายเงินผ่อนบ้าน โรงเรือน สิ่งปลูกสร้าง แพ":0,
  "รายได้จากการให้เช่าทรัพย์สิน การผิดสัญญาเช่าซื้อ การผิดสัญญาซื้อขายเงินผ่อน ที่ดินที่ใช้ในการเกษตร":0,
  "รายได้จากการให้เช่าทรัพย์สิน การผิดสัญญาเช่าซื้อ การผิดสัญญาซื้อขายเงินผ่อน ที่ดินที่มิได้ใช้ในการเกษตร":0,
  "รายได้จากการให้เช่าทรัพย์สิน การผิดสัญญาเช่าซื้อ การผิดสัญญาซื้อขายเงินผ่อน ยานพาหนะ":0,
  "รายได้จากการให้เช่าทรัพย์สิน การผิดสัญญาเช่าซื้อ การผิดสัญญาซื้อขายเงินผ่อน ทรัพย์สินอื่น":0,
  "รายได้จากวิชาชีพอิสระ ประกอบโรคศิลปะ": 0,
  "รายได้จากวิชาชีพอิสระฎหมาย วิศวกรรม สถาปัตยกรรม บัญชี ประณีตศิลปกรรม": 0,
  "รายได้จากรับเหมาก่อสร้าง": 0,
  "อื่นๆ": 0,
  },

  
# 0 mean no 1 mean yes 
  "ค่าลดหย่อน": {
    "คู่สมรสไม่มีเงินได้": 0,
    "ผู้มีเงินได้หรือคู่สมรสต่างฝ่ายต่างมีเงินได้": 0,
    "บุตร": {"คนที่1": {"อายุ":20 , "สถานะ":"ชอบด้วยกฎหมาย", "ระดับการศึกษา":"มหาลัย" ,"คนไร้ความสามารถหรือเสมือนไร้ความสามารถ":0},
                        "คนที่2": {"อายุ":20 , "สถานะ":"บุญธรรม" , "ระดับการศึกษา":"มหาลัย" ,"คนไร้ความสามารถหรือเสมือนไร้ความสามารถ":0}},
    "เลี้ยงดูบิดามารดาที่อายุ 60 ปีขึ้นไป": { "บิดา": {"ของตน": 0, "เงินได้ประจำปี":10000} , "มารดา":{"ของคู่สมรส":0,"เงินได้ประจำปี":5000}}
    "ค่าอุปการะผู้พิการ/ทุพพลภาพ": 0,
    "ค่าเบี้ยประกันชีวืต":0, 
    "ค่าประกันสุขภาพพ่อแม่": 0,
    "เงินสะสมกองทุนสำรองเลี้ยงชีพ": 0,
    "ค่าซื้อหน่วยลงทุนในกองทุนรวมเพื่อการเลี้ยงชีพ (RMF)": 0,
    "ค่าเบี้ยประกันชีวิตแบบบำนาญ:"0,
    "เงินสะสมกองทุนการออมแห่งชาติ":0,
    "ค่าซื้อหน่วยลงทุนในกองทุนรวมเพื่อการออม (SSF)":0 ,
    "ดอกเบี้ยกู้ยืม":0,
    "เงินสมทบประกันสังคม":0,
    "ค่าเบี้ยประกันสุขภาพ":0,
    "เงินบริจาค":0,
    "ค่าซื้อและค่าติดตั้งระบบกล้องโทรทัศน์วงจรปิด":0,
    "ค่าฝากครรภ์และค่าคลอดบุตร":0,
    "เงินลงทุนในหุ้นหรือจัดตั้งบริษัทเพื่อสังคม":0,
    "ค่าซื้อสินค้าหรือค่าบริการ":0

    "อื่นๆ": 0
  },
  "อื่นๆ": {
    "ภาษีหัก ณ ที่จ่าย": 0,
    "เครดิตภาษีเงินปันผล": 0,
     ...,
    "อื่นๆ": 0,
  }
}
```

In json structure in "ค่าลดหย่อน" I want you to decision to assign value if 0 mean don't have if 1 is mean have in json.
Removing any categories or subcategories in json with zero values except for "อื่นๆ". Ensure all calculations and data entries comply with the latest Thai tax regulations for the 2023 tax year (B.E. 2566).

Final JSON object structure:
```
{
  "เงินได้พึงประเมิน (ต่อปี)": {
   "เงินได้ประเภท 1-2": 500000,
    "อื่นๆ": 0,
  },
  "ค่าใช้จ่าย": {
    "อื่นๆ": 0,
  },
  "ค่าลดหย่อน": {
    "ส่วนตัว": 60000,
    "อื่นๆ": 0,
  },
  "อื่นๆ": {
    "อื่นๆ": 0,
  }
}
```




So if doesn't have input you choose delete that out 
Please return a JSON object without any description with four main categories:
"""


def calculate_thai_income_tax(income_data):
    output = ""

    output += "\nขั้นตอนการคำนวณภาษีเงินได้บุคคลธรรมดา ปี 2566:"

    # Parse JSON data
    data = json.loads(income_data) if isinstance(income_data, str) else income_data

    # 1. คำนวณเงินได้พึงประเมิน
    set_incomes = {}
    count = 1
    output += f"\n1. เงินได้พึงประเมิน (ต่อปี):"
    for i in data["เงินได้พึงประเมิน (ต่อปี)"]:
        # เงินได้พึงประเมินรวม
        if data["เงินได้พึงประเมิน (ต่อปี)"][i] != 0:
            if i == "เงินได้ประเภท 1-2":
                sum_1_2 = data["เงินได้พึงประเมิน (ต่อปี)"][i]

            set_incomes[i] = data["เงินได้พึงประเมิน (ต่อปี)"][i]
            output += f"\n1.{count} {i}: {data['เงินได้พึงประเมิน (ต่อปี)'][i]:,.2f} บาท"
            count += 1

    annual_income = sum(data["เงินได้พึงประเมิน (ต่อปี)"].values())
    output += f"\n    เหตุผล: รวมเงินได้ทุกประเภทตามที่ระบุในข้อมูล JSON"
    output += f"\n    Net money: {annual_income:,.2f} บาท"

    output += f"\n"
    output += f"\n2. ค่าใช้จ่าย:"
    # 2. คำนวณค่าใช้จ่าย
    if sum_1_2:
        expenses = min(sum_1_2 * 0.5, 100000)
        output += f"\n   หักค่าใช้จ่าย: {expenses} บาท"
        output += f"\n    เหตุผล: คำนวณค่าใช้จ่ายตามกฎหมาย คือ 50% ของเงินได้ประเภท 1-2 รวมกัน แต่ไม่เกิน 100,000 บาท"

        annual_income = annual_income - expenses
        output += f"\n   Net money after expenses: {annual_income:,.2f} บาท"

    if "ค่าแห่งกู๊ดวิลล์ ค่าแห่งลิขสิทธิ์หรือสิทธิอย่างอื่น" in set_incomes:
        expenses = min(set_incomes["ค่าแห่งกู๊ดวิลล์ ค่าแห่งลิขสิทธิ์หรือสิทธิอย่างอื่น"] * 0.5, 100000)
        output += f"\n   หักค่าใช้จ่าย: {expenses} บาท"
        output += f"\n    เหตุผล: คำนวณค่าใช้จ่ายตามกฎหมาย คือ 50% ของเงินได้ ค่าแห่งกู๊ดวิลล์ ค่าแห่งลิขสิทธิ์หรือสิทธิอย่างอื่น ไม่เกิน 100,000 บาท"
        annual_income = annual_income - expenses
        output += f"\n   Net money after expenses: {annual_income:,.2f} บาท"

    if "ดอกเบี้ย เงินปันผล ส่วนแบ่งกำไร ฯลฯ" in set_incomes:
        output += f"\n   หักค่าใช้จ่าย: 0 บาท"
        output += (
            f"\n    เหตุผล: ประเภทเงินได้ ดอกเบี้ย เงินปันผล ส่วนแบ่งกำไร ฯลฯ หักค่าใช้จ่ายไม่ได้"
        )
        output += f"\n   Net money after expenses: {annual_income:,.2f} บาท"

    list_5 = {
        "บ้าน โรงเรือน สิ่งปลูกสร้าง แพ": 0.3,
        "ที่ดินที่ใช้ในการเกษตร": 0.2,
        "ที่ดินที่มิได้ใช้ในการเกษตร": 0.15,
        "ยานพาหนะ": 0.30,
        "ทรัพย์สินอื่น": 0.10,
    }
    for i in list_5:
        if (
            f"รายได้จากการให้เช่าทรัพย์สิน การผิดสัญญาเช่าซื้อ การผิดสัญญาซื้อขายเงินผ่อน {i}"
            in set_incomes
        ):
            expenses = (
                set_incomes[
                    f"รายได้จากการให้เช่าทรัพย์สิน การผิดสัญญาเช่าซื้อ การผิดสัญญาซื้อขายเงินผ่อน {i}"
                ]
                * list_5[i]
            )
            output += f"\n   หักค่าใช้จ่าย: {expenses} บาท"
            output += f"\n    เหตุผล: ประเภทเงินได้จากการให้เช่าทรัพย์สิน การผิดสัญญาเช่าซื้อ การผิดสัญญาซื้อขายเงินผ่อน {i} {list_5[i]}%"
            annual_income = annual_income - expenses
            output += f"\n   Net money after expenses: {annual_income:,.2f} บาท"

    if "ดอกเบี้ย เงินปันผล ส่วนแบ่งกำไร ฯลฯ" in set_incomes:
        output += f"\n   หักค่าใช้จ่าย: 0 บาท"
        output += (
            f"\n    เหตุผล: ประเภทเงินได้ ดอกเบี้ย เงินปันผล ส่วนแบ่งกำไร ฯลฯ หักค่าใช้จ่ายไม่ได้"
        )
        output += f"\n   Net money after expenses: {annual_income:,.2f} บาท"

    if "รายได้จากวิชาชีพอิสระ ประกอบโรคศิลปะ" in set_incomes:
        expenses = set_incomes["รายได้จากวิชาชีพอิสระ ประกอบโรคศิลปะ"] * 0.6
        output += f"\n   หักค่าใช้จ่าย: {expenses} บาท"
        output += f"\n    เหตุผล: ประเภทเงินได้จากวิชาชีพอิสระ ประกอบโรคศิลปะ หักค่าใช้จ่าย 60%"
        annual_income = annual_income - expenses

        output += f"\n   Net money after expenses: {annual_income:,.2f} บาท"

    if "รายได้จากวิชาชีพอิสระฎหมาย วิศวกรรม สถาปัตยกรรม บัญชี ประณีตศิลปกรรม" in set_incomes:
        expenses = (
            set_incomes["รายได้จากวิชาชีพอิสระฎหมาย วิศวกรรม สถาปัตยกรรม บัญชี ประณีตศิลปกรรม"]
            * 0.3
        )
        output += f"\n   หักค่าใช้จ่าย: {expenses} บาท"
        output += f"\n    เหตุผล: ประเภทเงินได้จากวิชาชีพอิสระฎหมาย วิศวกรรม สถาปัตยกรรม บัญชี ประณีตศิลปกรรม หักค่าใช้จ่าย 30%"
        annual_income = annual_income - expenses

        output += f"\n   Net money after expenses: {annual_income:,.2f} บาท"

    if "รายได้จากรับเหมาก่อสร้าง" in set_incomes:
        expenses = set_incomes["รายได้จากรับเหมาก่อสร้าง"] * 0.6
        output += f"\n   หักค่าใช้จ่าย: {expenses} บาท"
        output += f"\n    เหตุผล: ประเภทเงินได้จาก 60%"
        annual_income = annual_income - expenses

        output += f"\n   Net money after expenses: {annual_income:,.2f} บาท"

    #### ยังไม่ทำอื่นๆ

    output += f"\n"

    # 4. คำนวณค่าลดหย่อน
    output += f"\n3. ค่าลดหย่อนรวม:"
    output += f"\n3.1 ค่าลดหยอนส่วนตัว"
    output += f"\n   เหตุผล: ค่าลดหยอนผู้มีเงินได้ 60,000 บาท"
    annual_income = annual_income - expenses
    output += f"\n   Net money after deductions: {annual_income:,.2f} บาท"

    # output +=data['ค่าลดหย่อน'])
    set_deductions = {}
    for i in data["ค่าลดหย่อน"]:
        if data["ค่าลดหย่อน"][i] != 0:
            set_deductions[i] = data["ค่าลดหย่อน"][i]

    # count_d = 1
    if "คู่สมรสไม่มีเงินได้" in set_deductions:
        output += f"\n ค่าลดหย่อน 60000 บาท"
        output += f"\n   เหตุผล: คู่สมรสไม่มีเงินได้สามารถลดหย่อนได้ 60,000 บาท"
        annual_income = annual_income - expenses
        output += f"\n   Net money after deductions: {annual_income:,.2f} บาท"

    ## ไม่เข้าใจ
    if "ผู้มีเงินได้หรือคู่สมรสต่างฝ่ายต่างมีเงินได้" in set_deductions:
        output += f"\n ค่าลดหย่อน 60000 บาท"
        output += f"\n   เหตุผล: คู่สมรสไม่มีเงินได้สามารถลดหย่อนได้ 60,000 บาท"
        annual_income = annual_income - expenses
        output += f"\n   Net money after deductions: {annual_income:,.2f} บาท"

    if "บุตร" in set_deductions and len(set_deductions["บุตร"]) > 1:
        count_child_1 = 0  # บุตรตามกฎหมาย
        count_child_2 = 0  # บุตรบุญธรรม

        for index, child in enumerate(list(set_deductions["บุตร"].keys())):
            if set_deductions["บุตร"][child]["สถานะ"] == "บุญธรรม":
                count_child_2 += 1

            if set_deductions["บุตร"][child]["สถานะ"] == "ชอบด้วยกฎหมาย":
                count_child_1 += 1

        deductions_child = count_child_1 * 30000
        output += f"\n ค่าลดหย่อน {deductions_child:,.2f} บาท"
        output += f"\n   เหตุผล: มีบุตรชอบตามกฎหมาย {count_child_1} คนทำให้สามารถลดย่อนได้ 30000 บาทต่อ 1 คนเป็น {deductions_child:,.2f} บาท"
        annual_income -= deductions_child
        output += f"\n   Net money after deductions: {annual_income:,.2f} บาท"

        if count_child_1 < 3:
            less = 3 - count_child_1

            deductions_child_2 = less * 30000
            if count_child_2 >= less:
                output += f"\n ค่าลดหย่อน {deductions_child_2:,.2f} บาท"
                output += f"\n   เหตุผล: มีบุตรบุญธรรม {count_child_2} คนทำให้สามารถลดย่อนได้ 30000 บาทต่อ 1 คนฉต่ไม่เกิน 3 คนรวมบุตรชอบตามกฎหมายเป็น {deductions_child_2:,.2f} บาท"
                annual_income -= deductions_child
                output += f"\n   Net money after deductions: {annual_income:,.2f} บาท"

    if "ค่าอุปการะผู้พิการ/ทุพพลภาพ" in set_deductions:
        output += f"\n ค่าลดหย่อน 60,000 บาท"
        output += (
            f"\n   เหตุผล: ค่าอุปการะเลี้ยงดูคนพิการหรือคนทุพพลภาพ หักค่าลดหย่อน คนละ 60,000 บาท"
        )
        annual_income -= deductions_child
        output += f"\n   Net money after deductions: {annual_income:,.2f} บาท"

    ### ค่าเบี้ยประกันชีวิต

    if "ค่าประกันสุขภาพพ่อแม่" in set_deductions:
        cost = min(set_deductions["ค่าประกันสุขภาพพ่อแม่"], 15000)
        output += f"\n ค่าลดหย่อน {cost} บาท"
        output += f"\n   เหตุผล: ค่าประกันสุขภาพของพ่อแม่ หักค่าลดหย่อนตามจริงไม่เกิน 15000 บาท"
        annual_income -= cost
        output += f"\n   Net money after deductions: {annual_income:,.2f} บาท"

    # 9
    if "เงินสะสมกองทุนสำรองเลี้ยงชีพ" in set_deductions:
        value = set_deductions["เงินสะสมกองทุนสำรองเลี้ยงชีพ"]
        if value > 10000 and value <= 490000:
            cost = value * 0.15 + 10000
            output += f"\n ค่าลดหย่อน {cost} บาท"

            ### แต่ไม่เกิน 490,000 บาท ซึ่งไม่เกินร้อยละ 15 ของค่าจ้างให้หักจากเงินได้ งงๆ

            output += f"\n   เหตุผล: เงินสะสมที่จ่ายเข้ากองทุนสำรองเลี้ยงชีพ หักค่าลดหย่อนตามจริงไม่เกิน 10000 บาท"
            annual_income -= cost
            output += f"\n   Net money after deductions: {annual_income:,.2f} บาท"

    # 11
    if "ค่าเบี้ยประกันชีวิตแบบบำนาญ" in set_deductions:
        cost = min(set_deductions["ค่าเบี้ยประกันชีวิตแบบบำนาญ"] * 0.15, 200000)
        output += f"\n ค่าลดหย่อน {cost} บาท"
        output += f"\n   เหตุผล: ค่าเบี้ยประกันชีวิตแบบบำนาญ หักค่าลดหย่อนในอัตราร้อยละ 15 แต่ไม่เกิน 200,000 บาทต่อปี"
        annual_income -= cost
        output += f"\n   Net money after deductions: {annual_income:,.2f} บาท"

    # if "เงินสะสมกองทุนการออมแห่งชาติ" in set_deductions:

    # 15
    if "เงินสมทบประกันสังคม" in set_deductions:
        cost = min(set_deductions["เงินสมทบประกันสังคม"], 200000)
        output += f"\n ค่าลดหย่อน {cost} บาท"
        output += f"\n   เหตุผล: เงินสมทบประกันสังคม หักค่าลดหย่อนเท่าที่จ่ายจริงเงินได้พึงประเมินที่ได้รับซึ่งต้องเสียภาษีเงินได้ในปีนั้น แต่ไม่เกิน 200,000 บาท"
        annual_income -= cost
        output += f"\n   Net money after deductions: {annual_income:,.2f} บาท"

    # 16
    # if "ค่าเบี้ยประกันสุขภาพ" in set_deductions:

    # 18
    # if "ค่าซื้อและค่าติดตั้งระบบกล้องโทรทัศน์วงจรปิด" in set_deductions:

    # 19
    if "ค่าฝากครรภ์และค่าคลอดบุตร" in set_deductions:
        cost = min(set_deductions["ค่าฝากครรภ์และค่าคลอดบุตร"], 60000)
        output += f"\n ค่าลดหย่อน {cost} บาท"
        output += f"\n   เหตุผล: ค่าฝากครรภ์และค่าคลอดบุตร หักค่าลดหย่อนเท่าที่จ่ายจริง แต่ละคราวไม่เกิน 60,000 บาท"
        annual_income -= cost
        output += f"\n   Net money after deductions: {annual_income:,.2f} บาท"

    ## 20
    if "เงินลงทุนในหุ้นหรือจัดตั้งบริษัทเพื่อสังคม" in set_deductions:
        cost = min(set_deductions["เงินลงทุนในหุ้นหรือจัดตั้งบริษัทเพื่อสังคม"], 100000)
        output += f"\n ค่าลดหย่อน {cost} บาท"
        output += f"\n   เหตุผล: เงินลงทุนในหุ้น หรือการเป็นหุ้นส่วนเพื่อจัดตั้ง หรือเพิ่มทุนบริษัท หรือห้างหุ้นส่วนนิติบุคคลที่ได้รับจดทะเบียนวิสาหกิจเพื่อสังคมและได้จดแจ้งการเป็นวิสาหกิจเพื่อสังคม หักลดหย่อนได้เท่าที่จ่ายจริง แต่ไม่เกิน 100,000 บาท"
        annual_income -= cost
        output += f"\n   Net money after deductions: {annual_income:,.2f} บาท"

    # ## 21
    # if "ค่าซื้อสินค้าหรือค่าบริการ" in set_deductions:
    #     cost = min(set_deductions['ค่าซื้อสินค้าหรือค่าบริการ'] ,30000)
    #     output +=f"\n ค่าหยดหย่อน {cost} บาท"
    #     output +=f"   เหตุผล: เป็นค่าซื้อสินค้าหรือบริการจากผู้ประกอบการจดทะเบียนภาษีมูลค่าเพิ่มและได้รับใบกำกับภาษี ค่าซื้อหนังสือ ค่าบริการหนังสือในรูปของข้อมูลอเล็กทรอนิกส์ผ่านระบบอินเทอร์เน็ต (e-books) ค่าซื้อสินค้า 1 ตำบล 1 ผลิตภัณฑ์ (OTOP)"
    #     annual_income -= cost
    #     output +=f"   Net money after deductions: {annual_income:,.2f} บาท"

    output += f"\n"

    # 5. คำนวณเงินได้สุทธิ
    output += f"\n4. เงินได้สุทธิ: {annual_income:,.2f} บาท"
    output += f"\n"

    # 6. คำนวณภาษี
    tax_brackets = [
        (150000, 0),
        (300000, 0.05),
        (500000, 0.10),
        (750000, 0.15),
        (1000000, 0.20),
        (2000000, 0.25),
        (5000000, 0.30),
        (float("inf"), 0.35),
    ]

    tax = 0
    remaining_income = annual_income
    previous_bracket = 0
    output += "\n5. การคำนวณภาษีตามขั้นเงินได้:"
    for bracket, rate in tax_brackets:
        if remaining_income <= 0:
            break
        taxable_in_bracket = min(remaining_income, bracket - previous_bracket)
        tax += taxable_in_bracket * rate
        remaining_income -= taxable_in_bracket
        output += f"\n   ช่วงเงินได้ {previous_bracket:,.0f} - {bracket:,.0f} บาท: เสียภาษี {taxable_in_bracket * rate:,.2f} บาท (อัตรา {rate*100}%)"
        previous_bracket = bracket

    output += f"\nภาษีที่ต้องชำระ: {tax:,.2f} บาท"
    output += f"\n   เหตุผล: คำนวณภาษีตามอัตราก้าวหน้า โดยแบ่งเงินได้สุทธิเป็นขั้นตามตารางภาษี"

    # 7. หักภาษีหัก ณ ที่จ่าย
    # withholding_tax = data["อื่นๆ"]["ภาษีหัก ณ ที่จ่าย"]
    # final_tax = max(tax - withholding_tax, 0)
    # output +=f"\n7. ภาษีที่ต้องชำระจริง: {final_tax:,.2f} บาท"
    # output +=f"   เหตุผล: นำภาษีที่คำนวณได้มาหักด้วยภาษีหัก ณ ที่จ่าย ({withholding_tax:,.2f} บาท)"

    return output, tax


def is_json(my_string):
    try:
        json_object = json.loads(my_string)
    except ValueError as e:
        return False
    return True



# def typhoon_instruct_OPENAPI_complete(
#     system_prompt, user_prompt, temperature=0.5, max_tokens=400, checked_json=False
# ):

#     def is_json(my_string):
#         try:
#             json_object = json.loads(my_string)
#         except ValueError as e:
#             return False
#         return True

#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "{system_prompt}"),
#             ("human", "{input}"),
#         ]
#     )

#     completion = prompt | ChatOpenAI(
#         api_key="bGlnaHRuaW5nOlUyM3pMcFlHY3dmVzRzUGFy",
#         openai_api_base="https://kjddazcq2e2wzvzv.snova.ai/api/v1/chat/completion",
#         model="llama3-70b-typhoon",
#         temperature=temperature,
#         max_tokens=max_tokens,
#     )

#     output = completion.invoke(
#         {"system_prompt": system_prompt, "input": user_prompt}
#     ).content

#     if checked_json == False:
#         return output
#     else:
#         if is_json(output):
#             return output
#         else:
#             return None



def typhoon_instruct_OPENAPI_complete(
    system_prompt, user_prompt, temperature=0.5, max_tokens=3200, checked_json=False
):

    def is_json(my_string):
        try:
            json_object = json.loads(my_string)
        except ValueError as e:
            return False
        return True

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            ("human", "{input}"),
        ]
    )

    completion = prompt | ChatOpenAI(
        api_key="sk-VojBupnpGWNgPjgxgXzsV0D1hqbFJ1wMAbnwlDOGcsorIKMZ",
        openai_api_base="https://api.opentyphoon.ai/v1",
        model="typhoon-v1.5x-70b-instruct",
        temperature=temperature,
        max_tokens=max_tokens,
    )

    output = completion.invoke(
        {"system_prompt": system_prompt, "input": user_prompt}
    ).content

    if checked_json == False:
        return output
    else:
        if is_json(output):
            return output
        else:
            return None


def tax_calculator(user_input: str):
    result_json = typhoon_instruct_OPENAPI_complete(
        system_prompt=SYSTEM_MESSAGE_TYPHOON_JSON_TAX,
        user_prompt=user_input,
        temperature=0,
        max_tokens=2048,
        checked_json=True,
    )

    if isinstance(result_json, str) and is_json(result_json):
        output, final_tax = calculate_thai_income_tax(result_json)
        return {"result": output}
    else:
        return {"result": "Error in processing the input\n"}
