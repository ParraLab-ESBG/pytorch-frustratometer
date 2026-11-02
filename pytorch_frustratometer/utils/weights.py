import torch

states = ["highly", "neutral", "minimally",]
welltypes = ["short", "long", "water-mediated"]
modes = ["mutational", "configurational"]

#Observed weights for single residue weights [frust state]
single_frequencies = torch.FloatTensor([
    0.11596672172398712,
    0.49032785004198104,
    0.3937054282340319
])
single_weights = 1/single_frequencies
single_weights = single_weights / single_weights.sum() * len(single_weights)

#Observed weights [mode, welltype, frust state]
pair_frequencies = torch.FloatTensor([
    [
        #[0.0000, 0.0000, 0.0000],
        [0.1070, 0.5171, 0.3758],
        [0.0779, 0.4739, 0.4482],
        [0.1646, 0.6074, 0.2280]
    ],
    [
        #[0.0000, 0.0000, 0.0000],
        [0.1396, 0.5574, 0.3030],
        [0.0182, 0.7429, 0.2389],
        [0.0420, 0.6721, 0.2859]
    ]
])
pair_weights = 1/pair_frequencies
pair_weights = pair_weights / pair_weights.sum(dim=-1, keepdim=True) * pair_weights.size(-1)

frustration_thresholds = torch.FloatTensor([
    [-1,   0.55],  # [0] single-residue
    [-1,   0.78],  # [1] pair (mutational / configurational / mj)
    [0.8, -1.2],   # [2] smooth — descending signals inverted scale; use classify_frustration()
])


def classify_frustration(fi, thresholds):
    """Classify FI values into 0=highly, 1=neutral, 2=minimally frustrated.

    thresholds: 1-D tensor [low_cut, high_cut] OR an int mode index into
                frustration_thresholds (0=single, 1=pair, 2=smooth).
    Ascending row: standard searchsorted.
    Descending row (smooth): inverted scale — flips bounds and applies
    (2 - searchsorted) so high FI → 0 (highly), low FI → 2 (minimally).
    """
    if isinstance(thresholds, int):
        thresholds = frustration_thresholds[thresholds]
    if thresholds[0] > thresholds[1]:
        return 2 - torch.searchsorted(thresholds.flip(0), fi)
    return torch.searchsorted(thresholds, fi)

loss_weights = torch.FloatTensor([
    1.0, #Residue frustration index
    1.0, #Residue frustration class

    #Pair/Edge frustration must be down-weighted because that can be up to N^2
    0.4, #Mutational frustration index
    0.4, #Mutational frustration class

    0.4, #Configurational frustration index
    0.4, #Configurational frustration class

    # auxiliary
    0.05, #sequence loss to match Frame2Seq

    0.05, #Single consistency loss
    0.05, #Mutational consistency loss
    0.05, #Configurational consistency loss

    # boundary losses (zero when boundary_margin=0, active otherwise)
    0.1,  #Single boundary loss
    0.05, #Mutational boundary loss (N^2 down-weighted)
    0.05, #Configurational boundary loss (N^2 down-weighted)
])

def get_loss_weights(epoch):
    """
    Curriculum learning: gradually introduce tasks
    Epoch 0-5: Sequence only (learn representations)
    Epoch 5-15: Add single-residue frustration
    Epoch 15-20: Gradually add pairwise frustration
    Epoch 20+: Full multitask
    """
    epoch_loss_weights = loss_weights.clone()  # Use .clone() not .copy()

    if epoch < 5:
        # Train only sequence
        epoch_loss_weights[:6] = 0.0    # Zero out all frustration
        epoch_loss_weights[10:] = 0.0   # Zero out all boundary losses

    elif epoch < 15:
        # Add single-residue frustration only
        epoch_loss_weights[2:6] = 0.0   # Zero out pairwise (mutational + configurational)
        epoch_loss_weights[11:] = 0.0   # Zero out pairwise boundary losses

    elif epoch < 20:
        # Gradually add pairwise frustration and boundary losses together
        alpha = (epoch - 15) / 5  # 0 to 1 over 5 epochs
        epoch_loss_weights[2:6] = alpha * loss_weights[2:6]
        epoch_loss_weights[11:] = alpha * loss_weights[11:]

    # else: epoch >= 20, use full weights

    return epoch_loss_weights



def get_consistency_temp(epoch, temp_start=2.0, temp_end=0.5):
    """Anneal consistency loss temperature from soft to sharp boundaries.

    Higher temp = softer sigmoid boundaries (easier early training).
    Lower temp = sharper boundaries (precise index-class agreement later).
    Follows the single-residue curriculum: active from epoch 5, converged by epoch 20.
    """
    if epoch < 5:
        return temp_start
    elif epoch < 20:
        alpha = (epoch - 5) / 15
        return temp_start - (temp_start - temp_end) * alpha
    else:
        return temp_end
