import requests
import json
import time

ROOT = "http://localhost:8080/api"
COOKIES = dict(simudyneSessionID="simudyneConnectorClient")


def getNexusID(model):
    r = requests.get(f"{ROOT}/simulation/{model}", cookies=COOKIES)
    p = r.json()
    results = p["results"]

    nexusID = ""

    if len(results) == 0:
        print("Starting new Nexus")
        n = requests.post(f"{ROOT}/simulation/{model}", cookies=COOKIES)

        nexusID = n.json()["id"]
    else:
        nexusID = results[0]["id"]

    r = requests.get(f"{ROOT}/nexus/{nexusID}", cookies=COOKIES)
    started = r.json()["isStarted"]
    if started == False:
        print("Starting Nexus")
        nexusCommand(nexusID, "start")

    return nexusID


def nexusCommand(nexusID, command):
    r = requests.post(f"{ROOT}/nexus/{nexusID}/{command}", cookies=COOKIES)
    if r.status_code == 400:
        if command != "step":
            print(f"Command {command} failed, retrying.")

        time.sleep(0.01)
        r = nexusCommand(nexusID, command)
    return r


def getModelData(nexusID):
    return requests.get(f"{ROOT}/nexus/{nexusID}/data", cookies=COOKIES).json()


def stepModel(nexusID):
    nexusCommand(nexusID, "step")


def setInputs(nexusID, data):
    r = requests.post(
        f"{ROOT}/nexus/{nexusID}/setJSONValue", json={"value": json.dumps(data)})

    if r.status_code != 200:
        print("Setting inputs failed, retrying")
        time.sleep(0.02)
        setInputs(nexusID, data)


def runModelGetLastPeriod(model, time_steps, parameters):
    nexusID = getNexusID(model)

    nexusCommand(nexusID, "restart")
    setInputs(nexusID, parameters)
    nexusCommand(nexusID, "setup")

    for t in range(0, time_steps):
        stepModel(nexusID)

    data = getModelData(nexusID)
    return data["data"][-1]["data"]


result = runModelGetLastPeriod("Game of Life", 100, {"gridSize": 300})
print(result["born"])
