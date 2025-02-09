import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from shadow_sg import scene
from shadow_sg.io import load_img
from shadow_sg.utils import compute_error_map, parse_args


@dataclass
class Args:
    path_to_exps: str = ""
    path_to_output: str = ""
    path_to_data_root: str = ""
    path_to_exp_root: str = ""
    img_height: int = 400 # 855 # human_signal # 600
    choose_best: bool = True
    choose_best_metric_name: str = "psnr"
    virtual_objs: bool = False
    img_err_scaled_factor: int = 8

def load_gt_img(path: Path) -> np.ndarray:
    img_gt = load_img(path)[..., :3]
    img_gt = ((img_gt ** (1 / 2.2)).clamp(0, 1) * 255).byte().detach().cpu().numpy()  
    return img_gt

def load_table(path_to_exp_root, path_to_data_root, exp_names,
    is_metrics: bool, virtual_objs: bool = False) -> list:
    table = []
    for row, exp_name in enumerate(exp_names):
        data_row = []

        path_to_data = Path(path_to_data_root) / exp_name
        path_to_exp = Path(path_to_exp_root) / exp_name

        # Find the path to the rendered image.
        path_to_iters = path_to_exp / "eval"
        if virtual_objs:
            path_to_iters = path_to_exp / "virtual_objs/eval"
        if path_to_iters.exists():  # SG
            path_to_iter = ""
            for p in path_to_iters.glob("*"):
                if str(p) > str(path_to_iter):
                    path_to_iter = p
            path_to_img = path_to_iter / "image.png"
        else:  # LSTSQ
            path_to_img = path_to_exp / "envmap/image.png"
            if virtual_objs:
                path_to_img = path_to_exp / "virtual_objs/envmap/image.png"

        if is_metrics:
            # Add metrics.
            with open(path_to_img.parent / "eval_plane.json") as fp:
                data_row.append(json.load(fp))
        else:
            # Add GT image.
            path_to_gt = path_to_data / "image.exr"
            if virtual_objs:
                path_to_gt = path_to_data / "image_virtual_objs.exr"
            if path_to_gt.exists():
                data_row.append(load_gt_img(path_to_gt))
            elif path_to_gt.with_suffix(".png").exists():
                img = load_gt_img(path_to_gt.with_suffix(".png"))
                data_row.append(img)
            else:
                data_row.append(None)
            if (path_to_gt.parent / "img_bboxes.png").exists():
                data_row[-1] = np.concatenate([
                    load_gt_img(path_to_gt.parent / "img_bboxes.png"),
                    data_row[-1]], axis=1)
            if (path_to_gt.parent / "image_original.png").exists():
                data_row[-1] = np.concatenate([
                    load_gt_img(path_to_gt.parent / "image_original.png"),
                    data_row[-1]], axis=1)

            # Add rendered image.
            data_row.append(np.array(Image.open(path_to_img)))

            # Add envmap.
            path_to_envmap = path_to_img.parent / "envmap.png"
            if path_to_envmap.exists():
                envmap = np.array(Image.open(path_to_envmap))
                if envmap.shape[1] == 4 * envmap.shape[0]:
                    envmap = np.concatenate([envmap, np.zeros_like(envmap)], axis=0)
                data_row.append(envmap)
            else:
                data_row.append(None)

            # Add error map.
            path_to_err_map = path_to_img.parent / "image_err.png"
            if path_to_err_map.exists():
                err_map = np.array(Image.open(path_to_err_map))
                data_row.append(err_map)
            else:
                data_row.append(None)

        table.append(data_row)
    return table

def choose_best(table: list, metrics_table: list, exp_names: list, metric_name = "psnr") -> tuple:
    ret_table = []
    ret_metrics_table = []
    for row, metrics_row in enumerate(metrics_table):
        ret_table_row = []
        ret_metrics_table_row = []
        ret_exp_names = []
        best_col = 0
        best_metric_val = 0
        for col, exp_name in enumerate(exp_names):
            if exp_name == "gt":
                continue
            metric_val = metrics_row[col]["img"][metric_name]
            if "sg" in exp_name:
                if metric_name == "psnr":
                    if metric_val > best_metric_val:
                        best_metric_val = metric_val
                        best_col = col
                elif metric_name == "lpips":
                    if metric_val < best_metric_val:
                        best_metric_val = metric_val
                        best_col = col
        for col, exp_name in enumerate(exp_names):
            if "sg" in exp_name and col != best_col:
                continue
            ret_table_row.append(table[row][col])
            ret_metrics_table_row.append(metrics_row[col])
            ret_exp_names.append(exp_name if not "sg" in exp_name else "sg")
        ret_table.append(ret_table_row)
        ret_metrics_table.append(ret_metrics_table_row)
    return ret_table, ret_metrics_table, ret_exp_names

def plot_visualization(path_to_output, image_table, height, exp_names):
    vis = []
    for row, exp_name in enumerate(exp_names):
        vis_row = []
        for col, img in enumerate(image_table[row]):
            if img is None:
                img = np.zeros((height, height, 3), dtype=np.uint8)
            else:
                h, w = img.shape[:2]
                img = cv2.resize(img, (int(height * w / h), height))
            vis_row.append(img)
        vis_row = np.concatenate(vis_row, axis=1)
        cv2.putText(vis_row, exp_name, (5, 60), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 255), 5)
        vis.append(vis_row)
    vis = np.concatenate(vis, axis=0)
    Image.fromarray(vis).save(path_to_output)

def plot_qualitative(path_to_output, image_table, image_err_table, envmap_table, width, height, scene_names, exp_names, img_err_scaled_factor):
    img_err_height, img_err_width = image_err_table[0][1].shape[:2]
    height = int(round(width * img_err_height / img_err_width))
    image_err_height = int(round(2 * height / (2 + img_err_width / img_err_height)))
    envmap_height = height - image_err_height
    envmap_width = 2 * envmap_height
    qualitative = []
    for row, scene_name in enumerate(scene_names):
        qualitative_row = []
        img_err_max = 0.0
        for img_err in image_err_table[row]:
            if img_err is not None:
                img_err_mean = img_err[img_err > 0].mean()
                img_err_std = img_err[img_err > 0].std()
                img_err_max = max(img_err_max, img_err_mean + img_err_scaled_factor * img_err_std)
        for col, exp_name in enumerate(exp_names):
            img = image_table[row][col]
            envmap = envmap_table[row][col]
            img = cv2.resize(img, (width, height))
            if exp_name == "gt":
                cell = img
                Image.fromarray(cell).save(path_to_output.parent / f"{scene_name}_{exp_name}.png")
                if envmap is not None:
                    Image.fromarray(envmap).save(
                        path_to_output.parent / f"{scene_name}_{exp_name}_envmap.png")
                    envmap = cv2.resize(envmap, (height // 2, height // 4))
                    # img[img.shape[0]-height//4:, :height//2] = envmap
                    # img[:height//4, :height//2] = envmap
                    Image.fromarray(img).save(path_to_output.parent / f"{scene_name}_{exp_name}_with_envmap.png")
            else:
                Image.fromarray(envmap).save(path_to_output.parent / f"{scene_name}_{exp_name}_envmap.png")
                envmap = cv2.resize(envmap, (envmap_width, envmap_height))
                img_err = image_err_table[row][col]
                img_err = compute_error_map(img_err, mask=(img_err != -1), max_val=img_err_max)
                img_err = (img_err * 255).astype(np.uint8)
                Image.fromarray(cv2.resize(img_err, (width, height))).save(path_to_output.parent / f"{scene_name}_{exp_name}_img_err.png")
                img_err = cv2.resize(img_err, (envmap_width, image_err_height))
                cell = np.concatenate([envmap, img_err], axis=0)
                cell = np.concatenate([img, cell], axis=1)
                Image.fromarray(img).save(path_to_output.parent / f"{scene_name}_{exp_name}_img.png")
                Image.fromarray(cell).save(path_to_output.parent / f"{scene_name}_{exp_name}_cell.png")
                # if col == len(exp_names) - 1:
                #     img_cb = draw_color_bar(0.0, img_err_max)
                #     img_cb = cv2.resize(img_cb, (int(round((img_cb.shape[1] / img_cb.shape[0] * height))), height))
                #     cell = np.concatenate([cell, img_cb], axis=1)
            qualitative_row.append(cell)
        qualitative_row = np.concatenate(qualitative_row, axis=1)
        Image.fromarray(qualitative_row).save(f"{Path(path_to_output).parent / Path(path_to_output).stem}_{scene_name}.png")
        qualitative.append(qualitative_row)
        print(f"{scene_name}:", img_err_max)
    qualitative = np.concatenate(qualitative, axis=0)
    Image.fromarray(qualitative).save(path_to_output)

def plot_metrics(path_to_output, metrics_table, img_type, metric_name, scene_names, exp_names):
    table = []
    for row, scene_name in enumerate(scene_names):
        table_row = [scene_name]
        for col, exp_name in enumerate(exp_names):
            if exp_name == "gt":
                continue
            table_row.append(f"{metrics_table[row][col][img_type][metric_name]:.2f}")
        table.append(table_row)
    table_headers = ["Scene"] + [exp_name for exp_name in exp_names if exp_name != "gt"]

    fig, ax = plt.subplots()
    ax.axis("off")
    plt_table = ax.table(cellText=table, colLabels=table_headers, loc="center")
    plt_table.auto_set_font_size(False)
    plt_table.set_fontsize(5)
    plt_table.auto_set_column_width(col=list(range(len(table_headers))))
    fig.savefig(path_to_output, bbox_inches="tight", dpi=300)
    plt.close(fig)

def generate_table_latex(path_to_output, metrics_table, scene_names, exp_names, env=False):
    # metric_names = list(metrics_table[0][1]["img"].keys())
    metric_names = ["psnr", "ssim", "lpips"]
    metric_display_names = list(metric_names)
    for i, metric_name in enumerate(metric_names):
        if metric_name in ["psnr", "ssim"]:
            metric_display_names[i] = metric_name.upper() + "$\\uparrow$"
        elif metric_name == "lpips":
            metric_display_names[i] = metric_name.upper() + "$\\downarrow$"
        elif metric_name == "scaled_rmse":
            metric_display_names[i] = "siRMSE$\downarrow$"
        elif metric_name == "ang_error":
            metric_display_names[i] = "Ang err.$\downarrow$"
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")
    if env:
        header1 = " & ".join(["", "", "\multicolumn{{{}}}{{c||}}{{Image}}".format(len(metric_display_names)),
            "\multicolumn{{{}}}{{c}}{{Environment map}}".format(len(metric_display_names))])
        header2 = " & ".join(["Illumination", "Method"] + metric_display_names + metric_display_names)
    else:
        header1 = " & ".join(["", "", "\multicolumn{{{}}}{{c}}{{Image}}".format(len(metric_display_names))])
        header2 = " & ".join(["Illumination", "Method"] + metric_display_names)
    
    header1 = "\\toprule " + header1 + "\\\\"
    header2 += "\\\\ \\midrule"

    latex = [header1, header2]
    baseline_method_names = {
        "lstsq_ss03": "\\ssPAMI",
        "lstsq_ss03_nnls": "\\ssPAMIN",
        "lstsq_sh21": "\\shICCV",
    }
    method_name_orders = {
        "\\ssPAMI": 0,
        "\\ssPAMIN": 1,
        "\\shICCV": 2,
        "Ours": 3,
    }
    metric_fmts = {
        "psnr": ".2f",
        "ssim": ".3f",
        "lpips": ".3f",
        "scaled_rmse": ".3f",
        "ang_error": ".3f",
    }
    for row, scene_name in enumerate(scene_names):
        latex_rows = ["" for _ in method_name_orders]
        for col, exp_name in enumerate(exp_names):
            if exp_name == "gt":
                continue
            if "sg" in exp_name:
                method_name = "Ours"
            else:
                method_name = baseline_method_names[exp_name]
            
            latex_row = []
            if method_name_orders[method_name] == 0:
                scene_vis_name = f"\\textsc{{{scene_name.split('_')[1].capitalize()}}}"
                latex_row.append(scene_vis_name)
            else:
                latex_row.append("")
            latex_row.append(method_name)
            for metric_name in metric_names:
                if method_name == "Ours":
                    latex_row.append(f"\\textbf{{{metrics_table[row][col]['img'][metric_name]:{metric_fmts[metric_name]}}}}")
                else:
                    latex_row.append(f"{metrics_table[row][col]['img'][metric_name]:{metric_fmts[metric_name]}}")
            if env:
                for metric_name in metric_names:
                    latex_row.append(f"{metrics_table[row][col]['env'][metric_name]:{metric_fmts[metric_name]}}")
            latex_row = " & ".join(latex_row)
            latex_row += " \\\\"
            if method_name_orders[method_name] == max(method_name_orders.values()):
                if row == len(scene_names) - 1:
                    latex_row += " \\bottomrule"
                else:
                    latex_row += " \\midrule"
            latex_rows[method_name_orders[method_name]] = latex_row
        latex += latex_rows

    with open(path_to_output, "w") as fp:
        fp.write("\n".join(latex))

def main():
    args: Args = parse_args(Args)

    Path(args.path_to_output).mkdir(parents=True, exist_ok=True)

    with open(args.path_to_exps) as fp:
        exp_names = fp.read().splitlines()

    image_table = load_table(args.path_to_exp_root, args.path_to_data_root,
        exp_names, False, args.virtual_objs)
    metrics_table = load_table(args.path_to_exp_root, args.path_to_data_root,
        exp_names, True, args.virtual_objs)

    # if args.choose_best:
    #     image_table = choose_best(image_table, metrics_table, exp_names, args.choose_best_metric_name)[0]
    #     envmap_table = choose_best(envmap_table, metrics_table, exp_names, args.choose_best_metric_name)[0]
    #     image_err_table, metrics_table, exp_names = choose_best(image_err_table, metrics_table, exp_names, args.choose_best_metric_name)

    plot_visualization(Path(args.path_to_output) / "visualization.jpg",
        image_table, args.img_height, exp_names)
    # plot_visualization(Path(args.path_to_output) / "envmap.png", envmap_table, args.img_width, args.img_height, scene_names, exp_names)
    # plot_qualitative(Path(args.path_to_output) / "qualitative.png", image_table, image_err_table, envmap_table,
    #     args.img_width, args.img_height, scene_names, exp_names, args.img_err_scaled_factor)

    # metric_names = list(metrics_table[0][1]["img"].keys())
    # for metric_name in metric_names:
    #     plot_metrics(Path(args.path_to_output) / f"image_{metric_name}.png", metrics_table, "img", metric_name, scene_names, exp_names)
    #     plot_metrics(Path(args.path_to_output) / f"envmap_{metric_name}.png", metrics_table, "env", metric_name, scene_names, exp_names)

    # generate_table_latex(Path(args.path_to_output) / "table_latex.txt", metrics_table, scene_names, exp_names)

if __name__ == "__main__":
    main()
