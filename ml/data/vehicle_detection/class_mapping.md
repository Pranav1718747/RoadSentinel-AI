# Class Mapping Specification — RoadSentimentalAI Vehicle Detection

**Dataset Source:** `daudshah/vehicle-detection-dataset` (Roboflow AI Traffic System v1)  
**Target Path:** `ml/data/vehicle_detection/`

---

## 1. Class Taxonomy

The downloaded dataset contains 7 distinct vehicle categories. To preserve clean semantic annotations without distorting model bounding box heads, we maintain all 7 classes explicitly in the fine-tuning pipeline:

| Original Class Index | Original Class Name | Mapped Target Index | Mapped Class Name | Primary Pipeline Category |
|:---:|---|:---:|---|---|
| **0** | `bicycle` | **0** | `bicycle` | Non-motorized / Light Transport |
| **1** | `bus` | **1** | `bus` | **Target Heavy Transport** |
| **2** | `car` | **2** | `car` | **Target Passenger Vehicle** |
| **3** | `motorbike` | **3** | `motorbike` | Two-Wheeler Motorized |
| **4** | `rickshaw` | **4** | `rickshaw` | Three-Wheeler Local Transport |
| **5** | `truck` | **5** | `truck` | **Target Freight Transport** |
| **6** | `van` | **6** | `van` | Passenger / Light Commercial |

---

## 2. Policy & Design Decisions

1. **NO Silent Merging**: `van` and `rickshaw` are retained as separate classes rather than silently merged into `car`. Silent merging dilutes bounding box regression precision because vans have distinct aspect ratios.
2. **Downstream Mapping**: In downstream RoadSentimentalAI modules (`track.py` / `track_v2.py`), `car`, `van`, `bus`, and `truck` are mapped to tracked vehicle entities.
3. **Reproducibility**: `ml/data/vehicle_detection/data.yaml` defines the 7 classes explicitly soUltralytics training resolves exact class indices.
