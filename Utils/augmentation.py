from CODE.Utils.package import *


class Augmentation:
    Avoid_name = ["get_method", "get_index"]

    def __init__(self) -> None:
        self.methods = {
            name: method
            for name, method in inspect.getmembers(self, predicate=inspect.isfunction)
        }

    @staticmethod
    def Jitter(x, p=0.9, amplitude=0.2, seed=None):
        if seed is not None:
            torch_gen = torch.Generator()
            torch_gen.manual_seed(seed)
            np_random_state = np.random.RandomState(seed)
        else:
            torch_gen = None
            np_random_state = np.random

        sign_array = (
            torch.randint(0, 2, size=x.shape, generator=torch_gen) * 2 - 1
        ).to(x.device)
        binary_array = (
            (torch.rand(x.shape, generator=torch_gen) < p).float().to(x.device)
        )
        spike_array = sign_array * binary_array * amplitude
        x_jitter = x + spike_array

        return x_jitter.to(x.device)

    @staticmethod
    def JitterWithDecay(x, p=0.5, amplitude=0.2, seed=None):
        if seed is not None:
            torch_gen = torch.Generator()
            torch_gen.manual_seed(seed)
            np_random_state = np.random.RandomState(seed)
        else:
            torch_gen = None
            np_random_state = np.random

        sign_array = (
            torch.randint(0, 2, size=x.shape, generator=torch_gen) * 2 - 1
        ).to(x.device)
        binary_array = (
            (torch.rand(x.shape, generator=torch_gen) < p).float().to(x.device)
        )
        spike_array = sign_array * binary_array * amplitude

        decay_factor = torch.linspace(1, 0, x.shape[-1]).to(x.device)
        decay_factor = decay_factor.expand_as(x)

        decayed_spike_array = spike_array * decay_factor
        x_jittered = x + decayed_spike_array

        return x_jittered.to(x.device)

    @staticmethod
    def binomial_mask(x, keep_prob=0.9, seed=None):
        if seed is not None:
            np_random_state = np.random.RandomState(seed)
        else:
            np_random_state = np.random

        mask = torch.from_numpy(
            np_random_state.binomial(1, keep_prob, size=x.shape)
        ).to(torch.bool)
        masked_x = x * mask.float().to(x.device)

        return masked_x

    @staticmethod
    def continuous_mask(x, max_chunk_ratio=0.05, overall_mask_ratio=0.25, seed=None):
        if seed is not None:
            torch_gen = torch.Generator()
            torch_gen.manual_seed(seed)
        else:
            torch_gen = None

        length = x.shape[-1]
        max_mask_length = int(max_chunk_ratio * length)
        total_mask_length = int(overall_mask_ratio * length)

        masked_arr = x.clone()
        current_mask_length = 0

        while current_mask_length < total_mask_length:
            start = torch.randint(0, length, (1,), generator=torch_gen).item()
            mask_length = torch.randint(
                1, max_mask_length + 1, (1,), generator=torch_gen
            ).item()

            if start + mask_length > length:
                mask_length = length - start
            if current_mask_length + mask_length > total_mask_length:
                mask_length = total_mask_length - current_mask_length
            if len(x.shape) == 3:
                masked_arr[:, :, start : start + mask_length] = 0
            else:
                masked_arr[:, start : start + mask_length] = 0

            current_mask_length += mask_length

        return masked_arr

    @staticmethod
    def gaussian_noise(x, mean=0, std=0.1, seed=None):
        if seed is not None:
            torch_gen = torch.Generator(device=x.device).manual_seed(seed)
        else:
            torch_gen = None

        size = x.shape
        mean_tensor = torch.full(size, mean).float().to(x.device)
        noise_array = torch.normal(mean_tensor, std, generator=torch_gen).to(x.device)
        noised_x = x + noise_array

        return noised_x

    @staticmethod
    def gaussian_smooth(x, kernel_size=10, sigma=5, seed=None):
        if seed is not None:
            torch_gen = torch.Generator()
            torch_gen.manual_seed(seed)
        else:
            torch_gen = None

        assert (
            len(x.shape) == 3 and x.shape[1] == 1
        ), "Expected input shape: [batch_size, 1, sample_length]"

        gauss_kernel = torch.exp(
            -torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size) ** 2
            / (2 * sigma**2)
        )
        gauss_kernel = gauss_kernel / gauss_kernel.sum()

        gauss_kernel = gauss_kernel.view(1, 1, -1).to(x.device)

        padding_size = kernel_size // 2
        start_mean = x[:, :, :padding_size].mean(dim=-1, keepdim=True)
        end_mean = x[:, :, -padding_size:].mean(dim=-1, keepdim=True)

        start_padding = start_mean.expand(-1, 1, padding_size)
        end_padding = end_mean.expand(-1, 1, padding_size)

        padded_x = torch.cat([start_padding, x, end_padding], dim=-1)

        smoothed_x = F.conv1d(padded_x, gauss_kernel, padding=0)
        smoothed_x = 0.5 * (smoothed_x[:, :, :-1] + smoothed_x[:, :, 1:])

        return smoothed_x

    @staticmethod
    def nothing(x, seed=None):
        return x

    @staticmethod
    def get_method(model=None):
        if model == None:
            model = Augmentation
        return {
            name: method
            for name, method in inspect.getmembers(model, predicate=inspect.isfunction)
            if isinstance(model.__dict__.get(name, None), staticmethod)
            and (not name in Augmentation.Avoid_name)
        }

    @staticmethod
    def get_index():
        return list(Augmentation.get_method().keys())


if __name__ == "__main__":
    print(Augmentation.get_method())
