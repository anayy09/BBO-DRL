## TITLE OF THE INVENTION

BIO-INSPIRED ADAPTIVE TASK OFFLOADING SYSTEM FOR ENERGY-EFFICIENT IOT-EDGE-CLOUD HEALTHCARE CONTINUUM

## FIELD OF THE INVENTION

The present invention relates to health informatics and edge computing. More particularly, the invention relates to a system and method for adaptive task offloading in IoT-enabled remote patient monitoring networks. The system uses bio-inspired metaheuristic scheduling to balance energy use, response time, and privacy when deciding whether to process a task on the wearable, an edge gateway, a nearby fog node, or a cloud server.

## BACKGROUND OF THE INVENTION

Internet of Medical Things (IoMT) devices such as wearable sensors and smart implants generate continuous streams of patient data. Common examples include ECG, SpO2, temperature, and blood pressure readings.

In a cloud-centric setup, raw data is sent to centralized servers for processing. This can add delay and can also increase bandwidth use. For time-critical events, such delay can be unsafe.

On the other hand, doing all processing on a wearable device drains battery faster and is limited by compute and memory. Many devices also cannot run heavier models locally.

Existing systems often use fixed rules such as sending data every few minutes. That approach does not adapt to changes in network conditions or to changes in patient status.

So, there is a need for an adaptive control layer that can decide, in real time, where a given task should run based on patient urgency, energy limits, network delay, and privacy constraints.

## OBJECTS OF THE INVENTION

- The primary object of the present invention is to provide a swarm-intelligence based decision engine for task offloading in a multi-tier healthcare network.
- Another object of the invention is to reduce end-to-end delay for critical health alerts while extending battery life of patient-worn devices.
- Another object of the invention is to include a context-aware priority mechanism that assigns higher priority to patients showing vital sign anomalies.
- Another object of the invention is to improve privacy by keeping sensitive raw data at the edge when feasible and sending only protected data to the cloud.

## SUMMARY OF THE INVENTION

The invention discloses a bio-inspired adaptive task offloading system for IoT-edge-cloud healthcare setups. The system includes three main layers.

- **IoMT Perception Layer:** wearable sensors that collect patient vitals.
- **Intelligent Edge Gateway:** a local node such as a smartphone or bedside hub that runs the scheduler and pre-processing.
- **Cloud Analysis Layer:** centralized servers for long-term storage and heavier analytics.

The core component is a metaheuristic scheduler running on the edge gateway. For each incoming task, it evaluates multiple execution options, such as local execution, edge execution, fog offloading, and cloud offloading. It then selects a near-best option based on a cost function that accounts for predicted energy use, predicted latency, and privacy risk. The weights in the cost function change based on a patient Criticality Index (CI), so the system favors fast response for high CI and favors energy saving for low CI.

A privacy guard applies a data protection step before any transfer outside the local trust boundary. Depending on policy, this can include encryption, feature-only transmission, or privacy noise for aggregated statistics.

## BRIEF DESCRIPTION OF THE DRAWINGS

[Figure 1](figures\Figure1.png) illustrates the three-tier architecture of the IoT-edge-cloud healthcare continuum.

[Figure 2](figures\Figure2.png) depicts the flowchart of the bio-inspired scheduler selecting between local execution, edge or fog offloading, and cloud offloading.

[Figure 3](figures\Figure3.png) shows an example energy versus latency trade-off produced by the decision engine under different patient criticality levels.

## DETAILED DESCRIPTION OF THE INVENTION

System Overview

The system acts as a control plane between patient sensors and clinical servers. It runs on an edge gateway that has more compute than the wearable devices and stays close to the patient.

### A. Data Ingestion and Task Definition

IoMT devices generate data packets or tasks. A task may be simple, such as filtering a signal, or heavier, such as running an arrhythmia detector. The edge gateway receives the packet and attaches metadata such as timestamp, device ID, and task type.

### B. Criticality Index (CI) Module

The CI module assigns a score that reflects clinical urgency. In one embodiment, CI is computed using a ruleset based on vital thresholds and trend changes. In another embodiment, CI is computed using an anomaly score from a lightweight model running on the gateway.

Example: If ECG rhythm becomes irregular beyond a threshold, CI is set high. If the data is a routine temperature log with a stable trend, CI is set low.

### C. Cost Estimation

For each candidate execution node (Wearable, Edge Gateway, Fog Node, Cloud), the system estimates:

- 1. Energy cost on the wearable and gateway, based on compute cycles, radio use, and payload size.
  - End-to-end latency, based on network delay, queue time, and runtime on the node.
  - Privacy risk, based on whether raw data leaves the local trust boundary and on the applied protection method.

### D. Decision Function and Weight Update

The scheduler uses a cost function F for each candidate decision x:

The weight values , , and are set based on CI.

When CI is high, is increased so the scheduler favors low latency. When CI is low, and are increased so the scheduler favors battery saving and privacy.

### E. Bio-Inspired Metaheuristic Scheduler

The scheduler searches across execution routes using a swarm approach. In one embodiment, it uses an ant-colony based method (ACO). Each ant represents a candidate route for the task, such as Wearable to Edge, Wearable to Fog, or Wearable to Cloud.

The ants evaluate routes using the cost function F. Routes with lower cost deposit more pheromone. Over iterations, the selection probability shifts toward routes with better cost under the current CI setting.

In another embodiment, the system uses a particle swarm search method (PSO) or a whale-based search method (WOA) to search the decision space, where each agent represents a candidate offloading choice.

### F. Adaptive Privacy Guard

Before sending any payload to a node outside the local trust boundary, the privacy guard applies a protection step. In one embodiment, the guard converts raw sensor streams into feature vectors on the gateway and transmits only features. In another embodiment, it encrypts the payload using a selected encryption mode. In another embodiment, it adds privacy noise to aggregated statistics where allowed by policy.

The guard can also enforce rules such as: 'raw ECG never leaves the gateway' and 'only derived features may be stored on the cloud'.

### G. Fallback and Safety Handling

If the chosen node becomes unreachable, the system falls back to the next feasible option based on the current cost ranking. For high CI tasks, the default fallback is edge execution to keep response time low.

## ADVANTAGES OF THE INVENTION

- Reduces battery drain on wearables by shifting heavier tasks away from the device when safe to do so.
- Reduces delay for critical alerts by preferring edge or fog processing when patient status is urgent.
- Adapts to changing network conditions because the decision is re-evaluated for each task or time window.
- Improves privacy by applying protection rules before data leaves the local trust boundary.
- Scales to more patients and devices since the same decision logic can be applied across many tasks.

## INDUSTRIAL APPLICABILITY

The invention is applicable in:

- Hospitals and smart wards for monitoring patients outside intensive care.
- Home and elderly care setups where continuous monitoring is needed with limited bandwidth.
- Tele-health platforms that need to reduce network cost while keeping timely alerts.
- Emergency and disaster deployments where connectivity is unreliable.

## CLAIMS

- A system for adaptive task offloading in a healthcare network, comprising: an IoMT perception layer for acquiring patient data; an intelligent edge gateway configured to receive the patient data as one or more tasks; and a cloud analysis layer, wherein the intelligent edge gateway comprises (i) a criticality index module configured to compute a Criticality Index (CI) representing clinical urgency of a task, (ii) a cost estimator configured to estimate, for each of a set of candidate execution nodes, a predicted energy cost, a predicted end-to-end latency, and a privacy risk, (iii) a bio-inspired metaheuristic scheduler configured to select an execution node for the task from the set of candidate execution nodes based on a cost function that combines the predicted energy cost, the predicted end-to-end latency, and the privacy risk using weights set as a function of the CI, and (iv) a privacy guard configured to apply a data protection step before transmitting a payload to an execution node outside a local trust boundary.
- The system as claimed in claim 1, wherein the IoMT perception layer includes one or more wearable sensors selected from ECG, SpO2, temperature, blood pressure, glucose, and activity sensors.
- The system as claimed in claim 1, wherein the set of candidate execution nodes includes: wearable-device execution, edge-gateway execution, a fog compute node, and a cloud server.
- The system as claimed in claim 1, wherein the criticality index module computes the CI based on at least one of: threshold rules on vital signs, a trend change measure, or an anomaly score generated on the edge gateway.
- The system as claimed in claim 1, wherein the cost function is of the form , and wherein the values of , , and are set based on the CI.
- The system as claimed in claim 5, wherein for a higher CI the value of is increased relative to and so that the selected execution node favors lower latency.
- The system as claimed in claim 5, wherein for a lower CI the values of and are increased relative to so that the selected execution node favors lower wearable energy use and lower privacy risk.
- The system as claimed in claim 1, wherein the bio-inspired metaheuristic scheduler uses an ant-colony based method (ACO), and wherein pheromone updates are based on the cost function value for candidate offloading routes.
- The system as claimed in claim 1, wherein the bio-inspired metaheuristic scheduler uses a swarm method selected from a particle swarm search method (PSO) and a whale-based search method (WOA) to search candidate offloading decisions.
- The system as claimed in claim 1, wherein the privacy guard applies a protection step including at least one of: encryption, feature-only transmission, or privacy noise addition to aggregated statistics.
- A computer-implemented method for adaptive task offloading in a healthcare network, comprising: receiving a task generated from IoMT patient data at an edge gateway; computing a Criticality Index (CI) for the task; estimating, for each of a set of candidate execution nodes, a predicted energy cost, a predicted end-to-end latency, and a privacy risk; setting weights of a cost function based on the CI; selecting an execution node using a bio-inspired metaheuristic scheduler based on the cost function; applying a privacy protection step when the selected execution node is outside a local trust boundary; and executing the task at the selected execution node and transmitting an alert or result to a clinician interface.
- The method as claimed in claim 11, further comprising performing a fallback selection to a next feasible execution node when a selected execution node becomes unavailable, wherein for high CI tasks the fallback selection prefers edge execution.
- A non-transitory computer-readable medium storing instructions which, when executed by one or more processors, perform the method as claimed in claim 11.

## ABSTRACT

The present invention discloses a bio-inspired adaptive task offloading system for IoT-edge-cloud healthcare networks. The system uses a Criticality Index to reflect patient urgency and a metaheuristic scheduler on an edge gateway to choose where each task should run across wearable devices, edge or fog nodes, and cloud servers. The decision combines predicted energy, predicted latency, and privacy risk using weights that change based on patient status. A privacy guard applies protection steps before data leaves the local trust boundary. The system reduces battery drain on wearables while keeping fast response for urgent clinical events.