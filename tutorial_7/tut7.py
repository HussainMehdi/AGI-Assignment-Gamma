from crewai import Agent, LLM
import os
llm_model = LLM(
    model="ollama/hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest",
    base_url="http://localhost:11434"
)


reception_agent = Agent(
    role="Reception Agent",
    goal="Register patient and assign ID",
    backstory="Collects patient info like name, age, contact.",
    llm=llm_model
)

queue_agent = Agent(
    role="Queue Agent",
    goal="Call patients in order",
    backstory="Maintains and updates the queue dynamically.",
    llm=llm_model
)

evaluation_agent = Agent(
    role="Evaluation Agent",
    goal="Evaluate patient vitals",
    backstory="Records vitals accurately for doctor review.",
    llm=llm_model
)

doctor_agent = Agent(
    role="Doctor Agent",
    goal="Diagnose patient and provide treatment",
    backstory="Analyzes vitals and symptoms to recommend treatment.",
    llm=llm_model
)


pharmacy_agent = Agent(
    role="Pharmacy Agent",
    goal="Dispense medicine as prescribed",
    backstory="Checks stock and gives dosage instructions.",
    llm=llm_model
)

def generate_patient_id():
    counter_file = "patient_counter.txt"


    # create file if not exists
    if not os.path.exists(counter_file):
        with open(counter_file, "w") as f:
            f.write("0")


    # read counter
    with open(counter_file, "r") as f:
        counter = int(f.read().strip())


    # increment and save
    counter += 1
    with open(counter_file, "w") as f:
        f.write(str(counter))


    # return formatted id
    return f"{counter:03d}"

def update_patient_file(patient_id, agent_name, note):
    file_path = f"patient_{patient_id}.md"
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"## {agent_name}\n")
        f.write(f"{note}\n\n")

def reception_task(name, age, contact):
    patient_id = generate_patient_id()


    patient_info = {
        "id": patient_id,
        "name": name,
        "age": age,
        "contact": contact
    }


    prompt = f"Register a new patient with info: {patient_info}"
    response = reception_agent.kickoff(prompt)
    note = response


    print("Reception Agent response:", note)
    update_patient_file(patient_id, "Reception Agent", note)


    return patient_id



def queue_task(patient_id):
    prompt = f"Call patient {patient_id} from the waiting list"
    response = queue_agent.kickoff(prompt)
    note = response
    update_patient_file(patient_id, "Queue Agent", note)


def evaluation_task(patient_id, vitals):
    prompt = f"Record vitals for patient {patient_id}: {vitals}"
    response = evaluation_agent.kickoff(prompt)
    note = response
    update_patient_file(patient_id, "Evaluation Agent", note)



def doctor_task(patient_id, symptoms):
    prompt = f"Patient {patient_id} symptoms: {symptoms}. Provide diagnosis and prescription."
    response = doctor_agent.kickoff(prompt)
    note = response
    update_patient_file(patient_id, "Doctor Agent", note)


def pharmacy_task(patient_id, prescription):
    prompt = f"Dispense medicine for patient {patient_id}: {prescription}"
    response = pharmacy_agent.kickoff(prompt)
    note = response
    update_patient_file(patient_id, "Pharmacy Agent", note)


patient_info = {"id": "001", "name": "Farrukh", "age": 25, "contact": "03133i33222"}
vitals = {"BP": "120/80", "Weight": "70kg"}
symptoms = "headache, nausea"
prescription = "Paracetamol 500mg, twice a day"


patient_id = reception_task("Farrukh", 25, "03133i33222")
queue_task(patient_id)
evaluation_task(patient_id, vitals)
doctor_task(patient_id, symptoms)
pharmacy_task(patient_id, prescription)

