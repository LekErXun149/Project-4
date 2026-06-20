# Bin Weighting System 
Weighting different type of SKU to output the correct quantity for each SKU\
Done by: Team 5\
1. Gerald Goh Xuan Chee: Project Leader & Lead Software Engineer\
●	Project Management: Directed the 13-week development timeline, managed team task allocation, and ensured all system requirements met CEVA Logistics' standards.\
●	Computer Vision & AI (YOLOv8s): Took primary ownership of training the YOLOv8s model for SKU identification and configured the drivers for the dual-camera system.\
●	ROS2 Middleware Architecture: Implemented the modular ROS2 (Jazzy) node structure and the integration logic that synchronized vision data with weight inputs.\
2. Lek Er Xun: Lead Mechanical Designer & Sensing Developer\
●	Precision CAD Modeling: Utilized SolidWorks to design the bin structure (MM01) and performed interference checks to ensure a seamless physical assembly.\
●	Load Cell Software Development: Authored the firmware for weight sensing, including the calibration of the 4-load-cell array and the communication protocol for the ADC modules.\
●	Rapid Prototyping: Managed the 3D printing of custom sensor mounts and spacers to ensure exact mechanical tolerances.\
3. Tung Ming Xuan: Lead Validation & Fabrication Specialist\
●	System Testing & QA: Designed and executed the 50+ test cycles to verify the 1% relative accuracy and 95% SKU identification success rates.\
●	Mechanical Fabrication: Managed the manual construction and material preparation of the bin frame, ensuring the physical build matched the SolidWorks specifications.\
●	Dataset Management: Responsible for the SKU onboarding process, accurately recording the "ground truth" weights and visual profiles for the identification model.\
4. Loh Sook Xian Rosalind: Procurement & Electrical Assembly Lead\
●	Electrical Hardening: Executed the soldering of the HX711 amplifier modules and core circuitry to move the project from volatile jumper wires to a permanent, vibration-resistant setup.\
●	Financial Accountability: Managed the Bill of Materials (BOM) and direct cost tracking to ensure the project stayed within the $500.00 budget.\
●	Sustainability Assessment: Conducted the evaluation of material choices and system power efficiency for the final report.\
5. Alson Lim Chin Meng: Hardware Operations & Mechanical Support\
●	Hardware Environment Deployment: Responsible for the physical setup and OS configuration of the Raspberry Pi 4 to provide a stable platform for the system middleware.\
●	Power & Connectivity Management: Integrated the powered USB 3.0 hub to provide a consistent 5V supply to the cameras, preventing hardware resets.\
●	Thermal Management: Installed the active cooling system (heatsink and fan) to ensure the processor maintained performance during high-load operations.\
●	Fabrication Support: Assisted in the mechanical assembly and material cutting for the bin weighting structure.\
