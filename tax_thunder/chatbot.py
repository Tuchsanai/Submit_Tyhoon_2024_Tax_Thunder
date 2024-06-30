import chainlit as cl
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from typing import Dict, Any
from chat_agent import graph_builder
import time
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"


# Example
action_dict1 = {
    "action1": {
        "value": "1",
        "label": "นายสมรัก ครอบครัว มีเงินเดือน 60,000 บาท/เดือน มีภรรยาไม่มีเงินได้ บุตร 3 คน จ่ายค่าเล่าเรียนบุตร 100,000 บาท/ปี และเงินสะสมกองทุนสำรองเลี้ยงชีพ 5% ของเงินเดือน",
    },
    "action2": {
        "value": "2",
        "label": "นายสมพล พลังชีวิต มีเงินเดือน 70,000 บาท/เดือน รายได้จากการเป็นโค้ชส่วนตัว 400,000 บาท/ปี จ่ายค่าเบี้ยประกันสุขภาพ 40,000 บาท และมีบุตร 3 คน จงคำนวณภาษีที่ต้องชำระ",
    },
    "action3": {
        "value": "3",
        "label": "นางสาวสมใจ ใจเย็น มีรายได้จากการเป็นนักกฎหมาย 1,500,000 บาท/ปี บริจาคเงินให้องค์กรสิทธิมนุษยชน 180,000 บาท และจ่ายค่าเบี้ยประกันสุขภาพ 35,000 บาท จงคำนวณภาษีที่ต้องชำระ",
    },
    "action4": {
        "value": "4",
        "label": "นายสมชาย ชายชาญ มีเงินเดือน 92,000 บาท/เดือน ได้รับเงินปันผลจากสหกรณ์ 80,000 บาท/ปี จ่ายดอกเบี้ยเงินกู้บ้าน 120,000 บาท/ปี จงคำนวณภาษีที่ต้องชำระ",
    },
}

action_dict2 = {
    "action5": {
        "value": "5",
        "label": "ผมเป็น freelance ทำงานไม่เหมือนกันเลยแต่ละเดือนผมต้องคิดภาษียังไงครับ?",
    },
    "action6": {
        "value": "6",
        "label": "รายจ่ายที่ยังไม่แน่นอนเอามาคำนวนกำไรสุทธิได้ไหม?",
    },
}


async def display_actions():
    actions1 = [
        cl.Action(name=key, value=info["value"], label=info["label"])
        for key, info in action_dict1.items()
    ]

    actions2 = [
        cl.Action(name=key, value=info["value"], label=info["label"])
        for key, info in action_dict2.items()
    ]

    await cl.Message(
        content="ตัวอย่างคำถามเกี่ยวกับ : การคํานวณภาษีเงินได้บุคคลธรรมดา :", actions=actions1
    ).send()
    await cl.Message(
        content="ตัวอย่างคำถามเกี่ยวกับ : กฏหมายภาษี และ ความรู้ภาษี ", actions=actions2
    ).send()


@cl.on_chat_start
async def setup():
    thread_id = f"thread_{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id}}

    memory = AsyncSqliteSaver.from_conn_string(":memory:")
    graph = graph_builder.compile(checkpointer=memory)

    cl.user_session.set("llm_chain", graph)
    cl.user_session.set("config", config)

    await display_actions()


@cl.on_message
async def query_llm(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    config = cl.user_session.get("config")

    # Create a Chainlit streaming callback
    async def stream_callback(event: Dict[str, Any]):
        for value in event.values():
            if "messages" in value and value["messages"]:
                content = value["messages"][-1].content
                await cl.Message(content=content).send()

    # Stream the response
    async for event in llm_chain.astream(
        {"messages": [("user", message.content)]}, config=config
    ):
        await stream_callback(event)


@cl.action_callback("action1")
@cl.action_callback("action2")
@cl.action_callback("action3")
@cl.action_callback("action4")
@cl.action_callback("action5")
@cl.action_callback("action6")
@cl.action_callback("action7")
@cl.action_callback("action8")
async def on_action(action):
    action_dict = action_dict1 if action.name in action_dict1 else action_dict2
    message = action_dict[action.name]["label"]

    await cl.Message(content= "Question = "+ message ).send()

    llm_chain = cl.user_session.get("llm_chain")
    config = cl.user_session.get("config")

    # Create a Chainlit streaming callback
    async def stream_callback(event: Dict[str, Any]):
        for value in event.values():
            if "messages" in value and value["messages"]:
                content = value["messages"][-1].content
                await cl.Message(content=content).send()

    # Stream the response
    async for event in llm_chain.astream(
        {"messages": [("user", message)]}, config=config
    ):
        await stream_callback(event)
