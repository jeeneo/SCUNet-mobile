import os
import sys
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from network_scunet import SCUNetWrapper


def convert_model(model_path, output_path, config=[4, 4, 4, 4, 4, 4, 4], dim=64, input_resolution=256, in_nc=3):
    """Convert SCUNet to a PyTorch Mobile-friendly model."""
    try:
        # instantiate and load weights
        device = torch.device('cpu')  # Use CPU for mobile conversion
        print(f"Creating SCUNetWrapper with config={config}, dim={dim}, in_nc={in_nc}")
        model = SCUNetWrapper(in_nc=in_nc, config=config, dim=dim, input_resolution=input_resolution)
        
        # load state dict
        print(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        
        # check if the state dict is directly loadable or needs the "model." prefix removed
        if any(k.startswith('model.') for k in state_dict.keys()):
            print("State dict has 'model.' prefix, removing it for proper loading")
            # remove the "model." prefix if it exists in the state dict
            new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.model.load_state_dict(new_state_dict)
        else:
            # direct loading
            model.model.load_state_dict(state_dict)
        
        model.eval()
        print("Model loaded successfully")

        # create example inputs for tracing
        # ise 1 channel for greyscale, 3 for color
        example_input = torch.randn(1, in_nc, 256, 256)
        print("Running a test inference pass")
        with torch.no_grad():
            _ = model(example_input)
            print("Test inference successful!")

        # script the model
        print("Scripting the model...")
        scripted_model = torch.jit.script(model)
        print("Model scripted successfully")
        
        # optimize for mobile
        print("Optimizing for mobile...")
        optimized_model = optimize_for_mobile(scripted_model)
        print("Model optimized for mobile")
        
        # save using lite interpreter
        print(f"Saving model to {output_path}")
        optimized_model._save_for_lite_interpreter(output_path)
        print(f"Model saved to: {output_path}")

        # test the saved model
        print("Testing the saved model...")
        loaded_model = torch.jit.load(output_path)
        with torch.no_grad():
            test_output = loaded_model(example_input)
        print("Model successfully loaded and tested!")
        
        # print shape of test output
        print(f"Test output shape: {test_output.shape}")
        
        return True
    except Exception as e:
        print(f"Error during model conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':

    # default: color model
    model_path = os.path.join('model_zoo', 'scunet_color_real_gan.pth')
    output_path = os.path.join('model_zoo', 'scunet_color_real_gan_mobile.ptl')
    in_nc = 3

    # uncomment below for greyscale models
    # model_path = os.path.join('model_zoo', 'scunet_gray_15.pth')
    # output_path = os.path.join('model_zoo', 'scunet_gray_15_mobile.ptl')
    # in_nc = 1

    # model_path = os.path.join('model_zoo', 'scunet_gray_25.pth')
    # output_path = os.path.join('model_zoo', 'scunet_gray_25_mobile.ptl')
    # in_nc = 1

    # model_path = os.path.join('model_zoo', 'scunet_gray_50.pth')
    # output_path = os.path.join('model_zoo', 'scunet_gray_50_mobile.ptl')
    # in_nc = 1

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Run the conversion
    convert_model(model_path, output_path, in_nc=in_nc)
