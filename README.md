Author : Subhrajyoti Pradhan
		 John Cullen 
		 Hetvi Mehta
		 Rahul Kant
		 Arya Seyfouri
		 
Project Details : System which automatically determines if soldiers are in pain (e.g. wounded), in real-time.
				  This project is written in Python.
			      The data for this project is available at GAIVI.
				  The data has already been split into train, test, and validationsets.

Data hierarchy : 
					1.pain_classification
						1.1. Testing
							1.1.1. Pain
									- Image Data
							1.1.2. No Pain
									- Image Data
						1.2. Training
							1.2.1. Pain
									- Image Data
							1.2.2. No Pain
									- Image Data
						1.3. Validation
							1.3.1. Pain
									- Image Data
							1.3.2. No Pain
									- Image Data
									
Procedure : We use deep learning to classify the images as either pain or no-pain. (CNN is recommended for this). 
			The script prints a confusion matrix, classification accuracy, precision, recall, and binary F1 score as output.
			We also use OpenCV for manual data augmentation. 
			
			