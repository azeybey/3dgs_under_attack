import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import urllib
import argparse
from torchvision.transforms.functional import to_pil_image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

def gaussian_pot():
    # 1D Gaussian
    x_1d = np.linspace(-5, 5, 1000)
    y_1d = multivariate_normal.pdf(x_1d, mean=0, cov=1)

    # 2D Gaussian
    x_2d, y_2d = np.mgrid[-5:5:.01, -5:5:.01]
    pos = np.dstack((x_2d, y_2d))
    rv_2d = multivariate_normal([0, 0], [[1, 0], [0, 1]])
    z_2d = rv_2d.pdf(pos)

    # Mean and covariance matrix for a 3D Gaussian
    mean = np.array([0.0, 0.0, 0.0])
    cov = np.diag([1.0, 1.0, 1.0])
    # Parameters for the multivariate Gaussian
    mu_x = 0
    mu_y = 0
    mu_z = 0
    variance = 1

    # Create grid and multivariate normal
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    x,  y = np.meshgrid(x, y)
    # Define the Gaussian function
    def gaussian(x, y, sigma=2):
        return np.exp(-(x**2 + y**2) / (2*sigma**2))

    # Apply the Gaussian function to the grid
    z = gaussian(x, y)


    # Create a 2D Gaussian distribution from the grids
    d = np.sqrt(x**2 + y**2)
    sigma, mu = 1.0, 0.0
    g = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))

    # Plotting
    fig = plt.figure(figsize=(18, 6))

    # 1D Gaussian
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(x_1d, y_1d)
    ax1.set_title('1D Gaussian Distribution')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Probability Density')

    # 2D Gaussian
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(x_2d, y_2d, z_2d, cmap='viridis')
    ax2.set_title('2D Gaussian Distribution')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Probability Density')

    # 3D Gaussian
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    #ax3.plot_wireframe(x, y, z)
    ax3.set_xlabel('X axis')
    ax3.set_ylabel('Y axis')
    ax3.set_zlabel('Z axis')
    ax3.set_title('3D Gaussian Distribution')

    plt.tight_layout()
    plt.show()



def attack(im, it):
    img_PIL = Image.open("arch_project/" + im)
    image = transform(img_PIL).unsqueeze(0).to(device)

    image_original = image.clone()
    image_original_np = image_original.detach().cpu().numpy()

    criterion = torch.nn.CrossEntropyLoss()

    for step in range(it):
        image.requires_grad = True

        outputs, _ = model(image, text)
        probs = outputs.softmax(dim=-1).detach().cpu().numpy()

        loss = criterion(outputs, torch.tensor([target_label], device=device))
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            image -= learning_rate * image.grad.sign()

        plt.ion()
        
        if step % 10 == 0:
            print(f"Step={step}, Loss={loss.item()}")
            print("True label[" + true_label_name + "]="+ str(probs[0,true_label]) + "\n" +
                "Target label["+ target_label_name + "]="+ str(probs[0,target_label]))

            image_np = image.detach().cpu().numpy()
            
            plt.figure(figsize=(24, 8))
            plt.clf()
            plt.subplot(1, 3, 1)
            plt.imshow(unnormalize(image_original_np[0]))
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(unnormalize(image_np[0]))
            plt.title("Noisy Image")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            perp = image_np[0] - image_original_np[0]
            diff_min = np.min(perp)
            diff_max = np.max(perp)
            scaled_perp = (perp - diff_min) / (diff_max - diff_min)
            plt.imshow(unnormalize(scaled_perp))
            plt.title("Perturbation")
            plt.axis('off')

            plt.draw()
            plt.pause(0.001)
            plt.ioff()  
            plt.show()

        image.requires_grad = False

    image = to_pil_image(unnormalize(scaled_perp).permute(2,0,1))
    image = image.resize((1080, 1080))
    image.save("perp1C.jpeg")

def find_or_add_label(label, class_list):
    if label in class_list:
        return class_list.index(label)
    else:
        class_list.append(label)
        return len(class_list) - 1

def unnormalize(im):
    tensor = torch.tensor(im).float()
    tensor = tensor.clone().detach().unsqueeze(0)
    tensor.mul_(std).add_(mean)
    tensor = tensor.cpu().squeeze().permute(1, 2, 0)
    return tensor

def test(folder):
    data_list = os.listdir("arch_project/" + folder)
    data_list.sort()

    for data_path in data_list:
        image_path = os.path.join("arch_project/" + folder, data_path)
        img_PIL = Image.open(image_path)
        image = transform(img_PIL).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs, _ = model(image, text)
        probs = outputs.softmax(dim=-1)
        del outputs
        torch.cuda.empty_cache()

        # Get the top 5 predictions and their indices
        top5_probs, top5_inds = probs.topk(5, dim=-1)

        # Detach and move to CPU
        top5_probs = top5_probs.detach().cpu().numpy()
        top5_inds = top5_inds.detach().cpu().numpy()

        print(f"Image: {data_path}  Prediction 1: {imagenet_classes[top5_inds[0, 0]]}, {top5_probs[0, 0]:.5f} 2: {imagenet_classes[top5_inds[0, 1]]}, {top5_probs[0, 1]:.5f} 3: {imagenet_classes[top5_inds[0, 2]]}, {top5_probs[0, 2]:.5f} 4: {imagenet_classes[top5_inds[0, 3]]}, {top5_probs[0, 3]:.5f} 5: {imagenet_classes[top5_inds[0, 4]]}, {top5_probs[0, 4]:.5f}")


if __name__ == '__main__':

    import json
    import urllib.request
    from torchvision.models import resnet50, ResNet50_Weights

    # Load the pre-trained ResNet-50 model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Use the model's .meta attribute to get the class names
    class_names = model.meta["categories"]

    # Printing the class names
    for i, class_name in enumerate(class_names):
        print(f"Class {i}: {class_name}")


    parser = argparse.ArgumentParser()
    parser.add_argument("--true_label_name", type=str, default='white wall')
    parser.add_argument("--target_label_name", type=str, default='christmas tree')
    parser.add_argument("--image_name", type=str, default='WALL_1C.jpeg')
    parser.add_argument("--iterations", type=int, default=31)
    parser.add_argument("--learning_rate", type=int, default=0.01)
    args = parser.parse_args()

    learning_rate = args.learning_rate

    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    class_idx = json.load(urllib.request.urlopen(url))
    imagenet_classes = [class_idx[str(k)][1] for k in range(len(class_idx))]

    if torch.cuda.is_available():
        device = "cuda:1"
    else:
        device = "cpu"

    #model, transform = clip.load("RN50", device=device) 
    model, transform = clip.load("ViT-B/32", device=device)

    true_label_name = args.true_label_name
    target_label_name = args.target_label_name

    true_label = find_or_add_label(true_label_name, imagenet_classes)
    target_label = find_or_add_label(target_label_name, imagenet_classes)

    text = clip.tokenize(imagenet_classes).to(device)

    #   CLIP's normalization parameters
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    #attack(args.image_name, args.iterations)
    #test("test1B")

    #test("test1C/ResNet/45 degree")
    #test("test1C/ResNet/90 degree")
    #test("test1C/ResNet/135 degree")

    #test("test1C/VisionTransformer/45 degree")
    #test("test1C/VisionTransformer/90 degree")
    #test("test1C/VisionTransformer/135 degree")

    #gaussian_pot()

