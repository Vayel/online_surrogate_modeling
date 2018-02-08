import requests
import json
import time

ROOT = "http://localhost:8080/api"
COOKIES = dict(simudyneSessionID="testCache")


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
    response = requests.get(f"{ROOT}/nexus/{nexusID}/data", cookies=COOKIES)
    # print(response.text)
    jsonVal = response.json()
    # print(jsonVal)

    return jsonVal


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

def getYExample(model, time_steps, outputName, parameters):
    json = runModelGetLastPeriod(model, time_steps, parameters)
    # Filtering to only get outputs
    # json.pop('time', None)
    # for inputName in parameters.keys():
    #     json.pop(inputName, None)

    return {outputName: json[outputName]}

{'fitnessNaiveAgents': -0.5213220806043452,
 'fitnessRationalAgents': -0.9985394691398598,
 'fractionRationalAgents': 0.16326751891204855,
 'fractionNaiveAgents': 0.8367324810879515,
 'profitsRationalAgents': -0.9985394691398598,
 'profitsNaiveAgents': -0.5213220806043452,
 'supplySlope': 1.6841556292773072,
 'priceInTime': -0.041646571654704395,
 'cost': 1.6405656639158588,
 'demandIntercept': 0.9557057199854423,
 'beta': 3.4242556253825365,
 'w': 0,
 'demandSlope': 1.0494624104763592}

def evaluateModelOnInputs(modelName,time_steps, inputs, outputName):
    # print(inputs[0])
    # print(len(inputs))
    outputs = [getYExample(modelName,time_steps,outputName,i) for i in inputs]
    return outputs


parametersRange = {
  "beta" : 5,
  "demandIntercept": 0,
  "demandSlope": 0.5,
  "supplySlope": 1.35,
  "w": 0,
  "cost": 1,
  "fitnessRationalAgents": 0,
  "z": 0
}

if __name__ == "__main__":
    print(parametersRange)
    result = runModelGetLastPeriod("Brock & Hommes -- ABM version", 100, parametersRange)
    print(result)