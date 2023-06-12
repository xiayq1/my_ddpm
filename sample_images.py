import argparse
import torch
import torchvision
import script_utils
import os


def main():
    args = create_argparser().parse_args()
    device = args.device
    os.makedirs(args.save_dir, exist_ok=True)

    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        diffusion.load_state_dict(torch.load(args.model_path))

        if args.use_labels:
            for label in range(10):
                y = torch.ones(args.num_images // 10, dtype=torch.long, device=device) * label
                # samples = diffusion.sample(args.num_images // 10, device, y=y)
                sampling_timesteps = 50
                # samples = diffusion.sample(args.num_images // 10, device, y=y, sampling_timesteps=sampling_timesteps)
                samples = diffusion.sample_ddim(args.num_images // 10, device, y=y, sampling_timesteps=sampling_timesteps)

                for image_id in range(len(samples)):
                    image = ((samples[image_id] + 1) / 2).clip(0, 1)
                    torchvision.utils.save_image(image, f"{args.save_dir}/ddim_{sampling_timesteps}-{label}-{image_id}.png")
        else:
            samples = diffusion.sample(args.num_images, device)

            for image_id in range(len(samples)):
                image = ((samples[image_id] + 1) / 2).clip(0, 1)
                torchvision.utils.save_image(image, f"{args.save_dir}/{image_id}.png")
    except KeyboardInterrupt:
        print("Keyboard interrupt, generation finished early")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(num_images=100, device=device) # num_images=10000
    # defaults.update(script_utils.diffusion_defaults())
    defaults = dict(
        num_images=100, 
        learning_rate=2e-4,
        batch_size=128,
        iterations=800000,

        log_rate=5000,
        checkpoint_rate=5000,
        log_dir="./ddpm_logs",
        project_name=None,

        model_checkpoint="./ddpm_logs/",
        optim_checkpoint=None,

        schedule_low=1e-4,
        schedule_high=0.02,

        device=device,
    )
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='./ddpm_logs/label_None-ddpm-2023-06-07-09-03-iteration-50000-model.pth')
    parser.add_argument("--save_dir", type=str,  default='./gene_samples')
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()