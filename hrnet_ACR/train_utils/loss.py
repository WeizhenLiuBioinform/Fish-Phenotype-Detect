import torch
import torch.nn as nn


class KpLoss_One(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        # [num_kps] -> [B, num_kps]
        kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])

        # [B, num_kps, H, W] -> [B, num_kps]
        loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])
        loss = torch.sum(loss * kps_weights) / bs
        return loss


class KpLoss(object):
    def __init__(self):
        self.criterion = nn.MSELoss(reduction='none')

        # Add learnable weights
        self.weight_main_loss = nn.Parameter(torch.ones(1))
        self.weight_additional_loss = nn.Parameter(torch.ones(1))
        # Ratio ranges for special keypoints
        self.kps_ratio_ranges = {
            0: (0.0, 0.0),
            1: (0.1685506154232605, 0.24185406149609912),
            2: (0.13334822425731516, 0.18618266978922718),
            3: (0.10975609756097561, 0.16186880604058518),
            4: (0.31323598681111636, 0.42519326966803095),
            5: (0.27379606601853945, 0.40177321511899206),
            6: (0.8139321723189734, 0.8985102420856611),
            7: (0.8276107032914989, 0.887921155168462),
            8: (1.0, 1.0),
            9: (0.84051329055912, 0.9079625292740047),
            10: (0.026697892271662763, 0.05583250249252243),
            11: (0.052407614781634936, 0.08549351944167498),
            12: (0.1576137418755803, 0.22372648003671408),
            13: (0.24815724815724816, 0.3786140431390546),
            14: (0.39566650965614697, 0.4727683615819209),
            15: (0.5034335780251007, 0.5902824858757062),
            16: (0.6208195949128592, 0.7043926490363066),
            17: (0.6982126489459212, 0.7943052391799544),
            18: (0.7727064220183486, 0.862709145083658),
            19: (0.3344324069712671, 0.445733623398598),
            20: (0.4884597268016957, 0.5968850206138342),
            21: (0.5103626943005182, 0.603298213467705)
        }
        self.y_ratios = {
            0: (0.3231884057971016, 0.8002754820936638),
            1: (0.3063328424153168, 0.5546334716459198),
            2: (0.2898368883312422, 0.0),
            3: (0.577127659574468, 0.9077134986225895),
            4: (0.06422018348623845, 0.01),
            5: (0.5075239398084817, 1.0),
            6: (0.027662517289073218, 0.7606382978723404),
            7: (0.2948557089084065, 0.7484662576687116),
            8: (1.0, 0.3562585969738652),
            9: (0.14167812929848705, 0.6279069767441859),
            10: (0.18010752688172058, 0.5523178807947021),
            11: (0.17180094786729866, 0.5576158940397352),
            12: (0.45668316831683153, 0.7326869806094183),
            13: (0.2568922305764409, 0.8243021346469622),
            14: (0.6930091185410334, 0.9548022598870057),
            15: (0.5775075987841944, 1.0),
            16: (0.6185567010309279, 0.974025974025974),
            17: (0.436637390213310, 0.8116883116883116),
            18: (0.49532710280373865, 1.0),
            19: (0.0, 0.07339449541284403),
            20: (0.013801756587201876, 0.2113207547169811),
            21: (0.0, 0.18679245283018858)
        }

    def to(self, device):
        """
        Move the learnable parameters of the loss to the specified device.
        """
        self.weight_main_loss.data = self.weight_main_loss.data.to(device)
        self.weight_additional_loss.data = self.weight_additional_loss.data.to(device)
        return self  # Return self to allow method chaining

    def compute_keypoint_positions(self, heatmap):
        """
        Determine the keypoint positions based on the maximum value in the heatmap.
        """
        # Ensure the input is a single heatmap
        assert len(heatmap.shape) == 2, 'Heatmap should be 2-ndim'

        height, width = heatmap.shape
        # Reshape the heatmap to find the maximum value
        heatmap_reshaped = heatmap.reshape(-1)
        maxval, idx = torch.max(heatmap_reshaped, dim=0)

        # If the maximum value is less than or equal to 0, return the center position
        if maxval <= 0:
            return torch.tensor([width / 2, height / 2])

        # Compute the coordinates
        x = idx % width
        y = torch.floor(idx / width)

        return torch.tensor([x.float(), y.float()])

    def apply_reverse_transformation(self, x, y, reverse_trans):
        # Ensure reverse_trans is a Tensor
        if not isinstance(reverse_trans, torch.Tensor):
            reverse_trans = torch.from_numpy(reverse_trans)
        # Convert coordinates to homogeneous coordinates and ensure the same data type as reverse_trans
        coords = torch.tensor([x, y, 1.0], dtype=reverse_trans.dtype)
        # Apply reverse transformation
        transformed_coords = torch.matmul(reverse_trans, coords)

        return transformed_coords[:2]

    def compute_additional_loss(self, logits, targets):
        additional_loss = 0
        bs, num_kps, _, _ = logits.shape

        for i in range(bs):
            # Get the true x and y coordinates of all keypoints in the current instance, divided by 4
            real_kps = targets[i]["keypoints"][:, :2] / 4.0  # Assume keypoint format is [x, y, visibility]

            # Get the reverse transformation matrix
            reverse_trans = targets[i]["reverse_trans"]

            # Transform all true keypoints back to the original coordinate system using the reverse transformation
            transformed_kps = []
            for kpt in real_kps:
                transformed_kpt = self.apply_reverse_transformation(kpt[0], kpt[1], reverse_trans)
                transformed_kps.append(transformed_kpt)

            transformed_kps = torch.stack(transformed_kps)

            # Compute the minimum and maximum x coordinates of all transformed keypoints
            min_x = transformed_kps[:, 0].min()
            max_x = transformed_kps[:, 0].max()
            # Get the minimum and maximum y values
            min_y = transformed_kps[:, 1].min()
            max_y = transformed_kps[:, 1].max()

            # Compute width
            width = max_x - min_x
            # Compute height
            height = max_y - min_y

            for k in self.kps_ratio_ranges:
                if k < num_kps:
                    heatmap = logits[i, k, :, :]
                    kps_x, kps_y = self.compute_keypoint_positions(heatmap)

                    # Apply reverse transformation
                    kps_x, kps_y = self.apply_reverse_transformation(kps_x, kps_y, reverse_trans)

                    # Get the ratio range for the keypoint
                    min_ratio, max_ratio = self.kps_ratio_ranges[k]
                    # Compute the allowed minimum and maximum x coordinates
                    allowed_min_x = min_x + min_ratio * width
                    allowed_max_x = min_x + max_ratio * width

                    # Compute the extent to which the keypoint is out of the allowed range
                    out_of_range = torch.clamp(allowed_min_x - kps_x, min=0) + torch.clamp(kps_x - allowed_max_x, min=0)

                    # Accumulate the out-of-range loss
                    additional_loss += out_of_range
            for k in self.y_ratios:
                if k < num_kps:
                    heatmap = logits[i, k, :, :]
                    kps_x, kps_y = self.compute_keypoint_positions(heatmap)

                    # Apply reverse transformation
                    kps_x, kps_y = self.apply_reverse_transformation(kps_x, kps_y, reverse_trans)

                    # Get the y ratio range for the keypoint
                    min_ratio_y, max_ratio_y = self.y_ratios[k]

                    # Compute the allowed minimum and maximum y coordinates
                    allowed_min_y = min_y + min_ratio_y * height
                    allowed_max_y = min_y + max_ratio_y * height

                    # Compute the extent to which the keypoint is out of the allowed y range
                    out_of_range_y = torch.clamp(allowed_min_y - kps_y, min=0) + torch.clamp(kps_y - allowed_max_y,
                                                                                             min=0)

                    # Accumulate the out-of-range y loss
                    additional_loss += out_of_range_y

        additional_loss /= (bs * len(self.kps_ratio_ranges))
        return additional_loss

    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4D'

        # Compute the main loss
        heatmaps = torch.stack([t["heatmap"] for t in targets]).to(logits.device)
        kps_weights = torch.stack([t["kps_weights"] for t in targets]).to(logits.device)
        original_loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])
        original_loss = torch.sum(original_loss * kps_weights) / logits.shape[0]

        # Compute the additional loss
        additional_loss = self.compute_additional_loss(logits, targets)

        # Weight the losses
        weighted_main_loss = self.weight_main_loss * original_loss
        weighted_additional_loss = self.weight_additional_loss * additional_loss

        # Total loss
        total_loss = weighted_main_loss + weighted_additional_loss
        return total_loss


class KpLoss_val(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')
        # Ratio ranges for special keypoints
        self.kps_ratio_ranges = {
            0: (0.0, 0.0),
            1: (0.1685506154232605, 0.24185406149609912),
            2: (0.13334822425731516, 0.18618266978922718),
            3: (0.10975609756097561, 0.16186880604058518),
            4: (0.31323598681111636, 0.42519326966803095),
            5: (0.27379606601853945, 0.40177321511899206),
            6: (0.8139321723189734, 0.8985102420856611),
            7: (0.8276107032914989, 0.887921155168462),
            8: (1.0, 1.0),
            9: (0.84051329055912, 0.9079625292740047),
            10: (0.026697892271662763, 0.05583250249252243),
            11: (0.052407614781634936, 0.08549351944167498),
            12: (0.1576137418755803, 0.22372648003671408),
            13: (0.24815724815724816, 0.3786140431390546),
            14: (0.39566650965614697, 0.4727683615819209),
            15: (0.5034335780251007, 0.5902824858757062),
            16: (0.6208195949128592, 0.7043926490363066),
            17: (0.6982126489459212, 0.7943052391799544),
            18: (0.7727064220183486, 0.862709145083658),
            19: (0.3344324069712671, 0.445733623398598),
            20: (0.4884597268016957, 0.5968850206138342),
            21: (0.5103626943005182, 0.603298213467705)
        }
        self.y_ratios = {
            0: (0.3231884057971016, 0.8002754820936638),
            1: (0.3063328424153168, 0.5546334716459198),
            2: (0.2898368883312422, 0.0),
            3: (0.577127659574468, 0.9077134986225895),
            4: (0.06422018348623845, 0.01),
            5: (0.5075239398084817, 1.0),
            6: (0.027662517289073218, 0.7606382978723404),
            7: (0.2948557089084065, 0.7484662576687116),
            8: (1.0, 0.3562585969738652),
            9: (0.14167812929848705, 0.6279069767441859),
            10: (0.18010752688172058, 0.5523178807947021),
            11: (0.17180094786729866, 0.5576158940397352),
            12: (0.45668316831683153, 0.7326869806094183),
            13: (0.2568922305764409, 0.8243021346469622),
            14: (0.6930091185410334, 0.9548022598870057),
            15: (0.5775075987841944, 1.0),
            16: (0.6185567010309279, 0.974025974025974),
            17: (0.436637390213310, 0.8116883116883116),
            18: (0.49532710280373865, 1.0),
            19: (0.0, 0.07339449541284403),
            20: (0.013801756587201876, 0.2113207547169811),
            21: (0.0, 0.18679245283018858)
        }

    def compute_keypoint_positions(self, heatmap):
        """
        Determine the keypoint positions based on the maximum value in the heatmap.
        """
        # Ensure the input is a single heatmap
        assert len(heatmap.shape) == 2, 'Heatmap should be 2-ndim'

        height, width = heatmap.shape
        # Reshape the heatmap to find the maximum value
        heatmap_reshaped = heatmap.reshape(-1)
        maxval, idx = torch.max(heatmap_reshaped, dim=0)

        # If the maximum value is less than or equal to 0, return the center position
        if maxval <= 0:
            return torch.tensor([width / 2, height / 2])

        # Compute the coordinates
        x = idx % width
        y = torch.floor(idx / width)

        return torch.tensor([x.float(), y.float()])

    def apply_reverse_transformation(self, x, y, reverse_trans):
        # Ensure reverse_trans is a Tensor
        if not isinstance(reverse_trans, torch.Tensor):
            reverse_trans = torch.from_numpy(reverse_trans)
        # Convert coordinates to homogeneous coordinates and ensure the same data type as reverse_trans
        coords = torch.tensor([x, y, 1.0], dtype=reverse_trans.dtype)
        # Apply reverse transformation
        transformed_coords = torch.matmul(reverse_trans, coords)

        return transformed_coords[:2]

    def compute_additional_loss(self, logits, targets):
        additional_loss = 0
        bs, num_kps, _, _ = logits.shape

        for i in range(bs):
            # Get the true x and y coordinates of all keypoints in the current instance, divided by 4
            real_kps = targets[i]["keypoints"][:, :2] / 4.0  # Assume keypoint format is [x, y, visibility]

            # Get the reverse transformation matrix
            reverse_trans = targets[i]["reverse_trans"]

            # Transform all true keypoints back to the original coordinate system using the reverse transformation
            transformed_kps = []
            for kpt in real_kps:
                transformed_kpt = self.apply_reverse_transformation(kpt[0], kpt[1], reverse_trans)
                transformed_kps.append(transformed_kpt)

            transformed_kps = torch.stack(transformed_kps)

            # Compute the minimum and maximum x coordinates of all transformed keypoints
            min_x = transformed_kps[:, 0].min()
            max_x = transformed_kps[:, 0].max()
            # Get the minimum and maximum y values
            min_y = transformed_kps[:, 1].min()
            max_y = transformed_kps[:, 1].max()

            # Compute width
            width = max_x - min_x
            # Compute height
            height = max_y - min_y

            for k in self.kps_ratio_ranges:
                if k < num_kps:
                    heatmap = logits[i, k, :, :]
                    kps_x, kps_y = self.compute_keypoint_positions(heatmap)

                    # Apply reverse transformation
                    kps_x, kps_y = self.apply_reverse_transformation(kps_x, kps_y, reverse_trans)

                    # Get the ratio range for the keypoint
                    min_ratio, max_ratio = self.kps_ratio_ranges[k]
                    # Compute the allowed minimum and maximum x coordinates
                    allowed_min_x = min_x + min_ratio * width
                    allowed_max_x = min_x + max_ratio * width

                    # Compute the extent to which the keypoint is out of the allowed range
                    out_of_range = torch.clamp(allowed_min_x - kps_x, min=0) + torch.clamp(kps_x - allowed_max_x, min=0)

                    # Accumulate the out-of-range loss
                    additional_loss += out_of_range
            for k in self.y_ratios:
                if k < num_kps:
                    heatmap = logits[i, k, :, :]
                    kps_x, kps_y = self.compute_keypoint_positions(heatmap)

                    # Apply reverse transformation
                    kps_x, kps_y = self.apply_reverse_transformation(kps_x, kps_y, reverse_trans)

                    # Get the y ratio range for the keypoint
                    min_ratio_y, max_ratio_y = self.y_ratios[k]

                    # Compute the allowed minimum and maximum y coordinates
                    allowed_min_y = min_y + min_ratio_y * height
                    allowed_max_y = min_y + max_ratio_y * height

                    # Compute the extent to which the keypoint is out of the allowed y range
                    out_of_range_y = torch.clamp(allowed_min_y - kps_y, min=0) + torch.clamp(kps_y - allowed_max_y,
                                                                                             min=0)

                    # Accumulate the out-of-range y loss
                    additional_loss += out_of_range_y

        additional_loss /= (bs * len(self.kps_ratio_ranges))
        return additional_loss

    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4D'

        # Compute the main loss
        heatmaps = torch.stack([t["heatmap"] for t in targets]).to(logits.device)
        kps_weights = torch.stack([t["kps_weights"] for t in targets]).to(logits.device)
        original_loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])
        original_loss = torch.sum(original_loss * kps_weights) / logits.shape[0]

        # Compute the additional loss
        additional_loss = self.compute_additional_loss(logits, targets)

        # Weight the losses
        weighted_main_loss = self.weight_main_loss * original_loss
        weighted_additional_loss = self.weight_additional_loss * additional_loss

        # Total loss
        total_loss = weighted_main_loss + weighted_additional_loss
        print("original_loss and additional_loss:", original_loss, "----", additional_loss)
        # total_loss = (1-m)*original_loss + m *additional_loss
        total_loss = original_loss + additional_loss
        print("total_loss:", total_loss)
        return original_loss.item(), additional_loss.item(), total_loss.item()

