from lavague.drivers.selenium import SeleniumDriver
from lavague.core import ActionEngine, WorldModel
from lavague.core.agents import WebAgent

# set up our three key components: driver, action engine, world model
driver = SeleniumDriver(headless=False)
action_engine = ActionEngine(driver)
world_model = WorldModel()

# create web agent
agent = WebAgent(world_model, action_engine)

# set url
agent.get("https://huggingface.co/docs")

# run agent with a specific objective
agent.run("Go on the quicktour of PEFT")
