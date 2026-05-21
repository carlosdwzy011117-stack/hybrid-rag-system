import sys
import os

#  scripts/ 下的脚本要 import src 里的东西，需要把项目根目录加进 sys.path
#    你 Day 3/6 在 scripts 下写脚本时做过同款（sys.path.insert）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.generator import Generator

#  造假数据：两篇假文档 + 一个假问题
fake_docs = [
    "猫是一种小型哺乳动物，通常作为宠物饲养。",
    "狗是人类最忠诚的朋友，喜欢摇尾巴。",
]
fake_query = "猫是什么动物？"

#  创建 Generator 实例
gen = Generator()

#  调 generate，传入 query 和 docs，拿到答案
answer = gen.generate(fake_query, fake_docs)

#  打印结果
print("=" * 40)
print("Query:", fake_query)
print("Answer:", answer)
print("=" * 40)