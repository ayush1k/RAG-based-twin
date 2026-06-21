# Projects

## Research Publications

### Intelligent Task Allocation in Autonomous Underwater Vehicle Swarms Using Ant Colony Optimization and Contract Net Protocol
* **Overview:** A research paper detailing task distribution architectures in decentralized multi-agent swarms of Autonomous Underwater Vehicles (AUVs).
* **Methodology:** 
  - Combined **Ant Colony Optimization (ACO)** (a metaheuristic optimization technique inspired by ant behavior to find optimal paths) with the **Contract Net Protocol (CNP)** (a structured negotiation framework where agents bid, award, and execute tasks).
  - Designed the hybrid system to dynamically allocate tasks among AUVs under communication and resource constraints, minimizing task completion time and swarm energy usage.
* **Significance:** Highlights expertise in agent-based negotiation, distributed coordination, and heuristic path-planning algorithms.

---

## Machine Learning Projects

## Financial Dashboard with AI Chatbot (Scikit-learn, Hugging Face, Flask, React)
* **Overview:** Full-stack dashboard serving financial visualizations and predictive metrics alongside a conversational chatbot assistant.
* **Key Implementations & Metrics:**
  - Built a predictive pricing engine using Linear Regression, achieving an **R² score of 0.85**.
  - Fine-tuned a **DialogGPT-medium** model, achieving a **90% relevance score** for financial chat queries.
  - Developed the end-to-end ML pipeline with a Flask API that serves real-time predictions (with model serialization) to a React frontend.

## AI Humanize GUI (Node.js, Docker, REST API, NLP, Render)
* **Overview:** A full-stack text rewriting application that converts machine-generated text into natural, human-like writing.
* **Key Implementations:**
  - Integrated a hybrid rule-based NLP engine and semantic AI to analyze and rewrite text.
  - Containerized the application workflow using **Docker** for local development and deployment reliability.
  - Deployed the service to **Render** via REST API endpoints for scalable cloud access.

## Eye Disease Classification (TensorFlow, EfficientNet, Scikit-learn)
* **Overview:** Deep learning classification system for identifying optical pathologies from retinal fundus images.
* **Key Implementations & Metrics:**
  - Fine-tuned **EfficientNetB3** with pre-trained ImageNet weights, achieving a **94% accuracy** on a 4-class eye disease dataset.
  - Applied data augmentation techniques that improved the minority class F1-score by **12%**.
  - Generated detailed evaluation metrics including confusion matrices and ROC curves.

## Leaf Detection using Transformers (PyTorch, Hugging Face, DETR, ResNet-50)
* **Overview:** Vision Transformer-based object detection system for identifying botanical leaves.
* **Key Implementations & Metrics:**
  - Fine-tuned Facebook's DEtection Transformer (**DETR**) with a **ResNet-50** backbone, achieving an **89% mAP** (mean Average Precision).
  - Optimized the training loop using **AdamW** optimizer with a learning rate of `1e-5`.
  - Integrated a custom collate function to handle variable-sized image batching using Hugging Face Transformers.

## Comparative Study of ML Algorithms (Scikit-learn, NumPy, Pandas)
* **Overview:** A systematic benchmark and evaluation of classical machine learning algorithms across multiple datasets.
* **Key Implementations & Metrics:**
  - Analyzed **10+ supervised and unsupervised algorithms** (including Linear/Logistic Regression, Decision Trees, SVM, KNN, K-Means, DBSCAN, and PCA) across 5 standard datasets.
  - Performed hyperparameter tuning using Grid Search and K-Fold Cross-Validation, resulting in **8% to 15% performance improvements**.

## Multi-Modal Agentic OCR & Document Intelligence Hub (React, Flask, Hugging Face)
* **Overview:** A document intelligence dashboard that combines a Flask backend, a **Florence-2** powered vision-language inference engine, and a React/Tailwind frontend.
* **Key Implementations:**
  - Uploads document images in-memory and processes them through the inference layer.
  - Returns extracted text and bounding boxes `[x1, y1, x2, y2]`.
  - Dynamically draws bounding overlays on a canvas layer matching the aspect ratio of the image.

## MedGemma-XRay (Jupyter Notebook, Python)
* **Overview:** Automated clinical reporting tool designed for chest X-ray report drafting.
* **Key Implementations:**
  - Leverages Google's **MedGemma** vision-language models to analyze chest X-ray findings and automatically generate structured findings and impressions.

## Urban Parking Dynamic Pricing System (Python, Pathway, Bokeh)
* **Overview:** A dynamic pricing engine that adjusts parking rates based on live occupancy streams and external factors, built for Summer Analytics 2025.
* **Key Implementations:**
  - Built baseline occupancy models (`Price = Price_{prev} + α * Occupancy/Capacity`).
  - Leveraged the **Pathway** streaming framework to process live occupancy feeds.
  - Ingested external event schedules and weather data streams to dynamically compute rate multipliers.
