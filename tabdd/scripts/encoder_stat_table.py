import pandas as pd

from tabdd.config.paths import RESULTS_CACHE_DIR

enc_stats = pd.read_csv(RESULTS_CACHE_DIR / f"encoder_stats.csv")

column = [
    "Model",
    "Recon. Acc. $\\uparrow$",
    "Recon. Acc. w/ FT $\\uparrow$",
    "FT Acc. $\\uparrow$",
    "\\# Enc. Params $\\downarrow$",
    "Dec. Params $\\downarrow$",
    "Clf. Params $\\downarrow$",
]

report = []
for model in ["MLP", "GNN", "TF"]:
    if model in ["GNN", "TF"]:
        one_line = f"{model} "
    elif model == "MLP":
        one_line = "FFN "
    vanilla = enc_stats[enc_stats["Model"] == model]
    ft = enc_stats[enc_stats["Model"] == f"{model}-MultiHead"]

    recon_acc = vanilla["Test Recon Accuracy"].quantile([0.25, 0.5, 0.75]).values
    recon_acc_ft = ft["Test Recon Accuracy"].quantile([0.25, 0.5, 0.75]).values
    clf_acc_ft = ft["Test Predict Accuracy"].quantile([0.25, 0.5, 0.75]).values
    enc_params = vanilla["Encoder Params"].quantile([0.25, 0.5, 0.75]).values
    dec_params = vanilla["Decoder Params"].quantile([0.25, 0.5, 0.75]).values
    clf_params = ft["Classifier Params"].quantile([0.25, 0.5, 0.75]).values

    report.append([
        ("FFN" if model == "MLP" else model),
        f"{{\\tiny {recon_acc[0]:.4f}}} {recon_acc[1]:.4f} {{\\tiny {recon_acc[2]:.4f}}}",
        f"{{\\tiny {recon_acc_ft[0]:.4f}}} {recon_acc_ft[1]:.4f} {{\\tiny {recon_acc_ft[2]:.4f}}}",
        f"{{\\tiny {clf_acc_ft[0]:.4f}}} {clf_acc_ft[1]:.4f} {{\\tiny {clf_acc_ft[2]:.4f}}}",
        f"{{\\tiny {enc_params[0]:.0f}}} {enc_params[1]:.0f} {{\\tiny {enc_params[2]:.0f}}}",
        f"{{\\tiny {dec_params[0]:.0f}}} {dec_params[1]:.0f} {{\\tiny {dec_params[2]:.0f}}}",
        f"{{\\tiny {clf_params[0]:.0f}}} {clf_params[1]:.0f} {{\\tiny {clf_params[2]:.0f}}}",
    ])

report = pd.DataFrame(report, columns=column)

print(report.to_latex(index=False, column_format="ccccccc"))
