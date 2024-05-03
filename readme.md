# Speech Separation + Noise cancellation

## Initial training method
ConvTasnetTraining.ipynb is the initial training method we used. We trained the model on a Sagemaker notebook instance. We encountered CUDA errors that say that the GPU is out of memory. 

## Sagemaker-specific training
We shifted our training method to a Sagemaker Training instance as we read that it was more memory efficient than using normal pytorch code on a Sagemaker notebook. Running SagemakerTraining.ipynb on a Sagemaker notebook instance will train the model.

Since we are using a p2.8xlarge instance and distributing the training across 8 GPUs, we assumed that the CUDA errors that say that the GPU is out of memory would not occur. However, we still encountered this error. This is the error we are trying to debug.

## Data
The data source for both training methods is stored in an S3 bucket so I will have to figure out how to share the data with you. I will update this readme with the data source.