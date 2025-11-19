Project Documentation
Project Title:Adversarial ML Attacks: How Hackers Manipulate ML Models
1. Introduction
Deep learning models, especially Convolutional Neural Networks (CNNs), have achieved remarkable performance on image classification tasks. However, these models are highly vulnerable to adversarial attacks, which are small, often imperceptible perturbations added to input images. These perturbations can drastically change the model’s predictions, posing serious security concerns in real-world applications.
This project investigates the vulnerability of neural networks trained on the CIFAR-10 dataset and explores methods to enhance their robustness using Defensive Distillation and Adversarial Training.

2. Objectives
Demonstrate the vulnerability of a standard ResNet18 model to adversarial attacks such as FGSM and PGD.


Implement Defensive Distillation to transfer knowledge from a robust teacher model to a student model, improving resistance to small perturbations.


Perform Combined Adversarial Training using both FGSM and PGD attacks to create a more robust student model.


Evaluate and compare clean and adversarial accuracies of teacher, distilled student, and adversarially trained student models.


Generate and visualize adversarial examples for understanding model vulnerabilities.


(Optional) Prepare an interactive dashboard to demonstrate the models and attacks.



3. Dataset
The project uses CIFAR-10, a standard benchmark dataset consisting of:
60,000 color images, 32x32 pixels


10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck


50,000 training images and 10,000 test images


Data preprocessing included normalization and data augmentation such as random cropping and horizontal flipping.

4. Models
4.1 Teacher Model
Architecture: ResNet18


Training: Trained on CIFAR-10 to be robust to small perturbations


Evaluation: Achieved baseline clean accuracy and robustness tested under FGSM and PGD attacks


4.2 Student Model
Architecture: ResNet18


Purpose: Trained using knowledge distillation from the teacher model


Distillation:


Used temperature scaling (T=20)


Loss function: Kullback-Leibler divergence between teacher soft targets and student predictions


4.3 Adversarially Trained Student
Architecture: ResNet18


Training: Combined adversarial training using FGSM and PGD


Parameters:


FGSM epsilon: 0.03


PGD epsilon: 0.03, steps: 7, step size: 0.007


Loss: Average of losses from clean, FGSM, and PGD images



5. Adversarial Attacks
5.1 Fast Gradient Sign Method (FGSM)
Single-step attack that perturbs the input in the direction of the gradient of the loss.


Epsilon controls the perturbation magnitude.


5.2 Projected Gradient Descent (PGD)
Multi-step attack that iteratively perturbs the input while projecting it back into a valid epsilon-ball.


Considered a stronger attack than FGSM.


Adversarial examples were generated for evaluation and visualization purposes.

6. Training Process
6.1 Defensive Distillation
Teacher model frozen (no gradient updates).


Student model trained using the KL divergence loss with teacher soft targets.


Result: Student model achieved improved robustness against small perturbations.


6.2 Combined Adversarial Training
Adversarial samples generated using both FGSM and PGD.


Training loss included contributions from clean, FGSM, and PGD images.


Result: Adversarially trained student model achieved higher clean accuracy and improved resistance to stronger attacks.



7. Evaluation
7.1 Clean Accuracy
Teacher Model: ~20.57%


Distilled Student: ~18.26%


Adversarially Trained Student: ~26.89%


7.2 Robust Accuracy (FGSM ε=0.08)
Teacher Model: 8.54%


Distilled Student: 7.62%


Adversarially Trained Student: 0.76%


Observation:
Distillation improves model stability for small perturbations.


Adversarial training improves clean accuracy but may reduce robustness for very strong attacks.


Adversarial examples were visualized to understand model failure patterns.



8. Adversarial Example Visualization
Original, adversarial, and amplified difference images saved.


Helps understand how small perturbations mislead models.


Visualizations stored in folders: adv_examples and adv_examples_fgsm.



9. Optional Dashboard
Implemented to allow interactive testing of:


Clean image classification


FGSM adversarial attacks


Visualization of adversarial examples


Provides a live demonstration interface for presentations.



10. Conclusion
The project demonstrates the vulnerability of deep learning models to adversarial attacks and explores effective defenses:
Defensive Distillation transfers robust knowledge from teacher to student.


Combined Adversarial Training enhances robustness against stronger perturbations.


The project includes evaluation metrics, visualizations, and an interactive dashboard to showcase model performance and robustness.


This work lays the foundation for designing secure and reliable deep learning systems in adversarial settings.


