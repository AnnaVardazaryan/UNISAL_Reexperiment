"""
UNISAL Training and Evaluation Scripts

WandB Integration:
    To enable WandB logging, use the following parameters:
    - use_wandb=True: Enable WandB logging
    - wandb_project="your_project_name": Set WandB project name (default: "unisal")
    - wandb_entity="your_entity": Set WandB entity/username (optional)

Examples:
    # Regular training with WandB
    python run.py train --use_wandb=True --wandb_project="unisal_experiment"
    
    # Fine-tuning with WandB
    python run.py train_finetune_mit --use_wandb=True --wandb_project="unisal_finetune"
"""

from pathlib import Path
import os

import fire

from pathlib import Path
import os
import random
import re

import cv2
import numpy as np
import torch

import unisal


def train(eval_sources=('SALICON', 'UCFSports', 'DHF1K'),
          **kwargs):
    """Run training and evaluation."""
    trainer = unisal.train.Trainer(**kwargs)
    trainer.fit()
    for source in eval_sources:
        trainer.score_model(source=source)
        trainer.export_scalars()
        trainer.writer.close()

def train_finetune_mit(eval_sources=('MIT300',),
          train_id=None, pretrained_train_id=None, **kwargs):
    """Run fine-tuning on MIT1003 and evaluation.
    
    Args:
        eval_sources: Sources to evaluate after fine-tuning
        train_id: Train ID for the fine-tuning run (creates new training directory)
        pretrained_train_id: Train ID to load pretrained weights from. 
                           If None, uses current train_dir or pretrained_unisal as fallback.
        **kwargs: Additional arguments passed to Trainer
    """
    # If train_id is specified, use it as the prefix
    if train_id is not None:
        # Extract prefix and suffix from train_id
        # train_id format is usually "prefix_suffix" or "timestamp_suffix"
        parts = train_id.split('_', 1)
        if len(parts) == 2:
            kwargs['prefix'] = parts[0]
            kwargs['suffix'] = parts[1]
        else:
            kwargs['suffix'] = train_id
    
    trainer = unisal.train.Trainer(**kwargs)
    trainer.fine_tune_mit(pretrained_train_id=pretrained_train_id)
    for source in eval_sources:
        trainer.score_model(source=source)
        trainer.export_scalars()
        trainer.writer.close()

def load_trainer(train_id=None):
    """Instantiate Trainer class from saved kwargs."""
    if train_id is None:
        train_id = 'pretrained_unisal'
    print(f"Train ID: {train_id}")
    train_dir = Path(os.environ["TRAIN_DIR"])
    train_dir = train_dir / train_id
    return unisal.train.Trainer.init_from_cfg_dir(train_dir)


def score_model(
        train_id=None,
        sources=('DHF1K', 'SALICON', 'UCFSports', 'Hollywood', 'FINAL_TEST_MIT300'),
        **kwargs):
    """Compute the scores for a trained model."""

    trainer = load_trainer(train_id)
    for source in sources:
        trainer.score_model(source=source, **kwargs)


def generate_predictions(
        train_id=None,
        sources=('DHF1K', 'SALICON', 'UCFSports', 'Hollywood',
                 'MIT1003', 'MIT300'),
        **kwargs):
    """Generate predictions with a trained model."""

    trainer = load_trainer(train_id)
    for source in sources:

        # Load fine-tuned weights for MIT datasets
        if source in ('MIT1003', 'MIT300'):
            trainer.model.load_weights(trainer.train_dir, "ft_mit1003")
            trainer.salicon_cfg['x_val_step'] = 0
            kwargs.update({'model_domain': 'SALICON', 'load_weights': False})

        trainer.generate_predictions(source=source, **kwargs)


def predictions_from_folder(
        folder_path, is_video, source=None, train_id=None, model_domain=None):
    """Generate predictions of files in a folder with a trained model."""

    # Allows us to call this function directly from command-line
    folder_path = Path(folder_path).resolve()
    is_video = bool(is_video)

    trainer = load_trainer(train_id)
    trainer.generate_predictions_from_path(
        folder_path, is_video, source=source, model_domain=model_domain)


def predict_examples(train_id=None):
    for example_folder in (Path(__file__).resolve().parent / "examples").glob("*"):
        if not example_folder.is_dir():
            continue

        source = example_folder.name
        is_video = source not in ('SALICON', 'MIT1003')

        print(f"\nGenerating predictions for {'video' if is_video else 'image'} "
              f"folder\n{str(source)}")

        if is_video:
            if not example_folder.is_dir():
                continue
            for video_folder in example_folder.glob('[!.]*'):   # ignore hidden files
                predictions_from_folder(
                    video_folder, is_video, train_id=train_id, source=source)

        else:
            predictions_from_folder(
                example_folder, is_video, train_id=train_id, source=source)



def visualize_examples(
    folder_path,
    train_id,
    n=12,
    seed=0,
    out_dirname="viz_out",
    alpha=0.55,
    colormap=cv2.COLORMAP_JET,
    model_domain=None,
    show_labels=True,
):
    """
    Visualize (original image + image with GT fixation overlay + prediction with GT overlay).
    Works for both image and video examples.

    Expected structure for images:
      folder_path/
        ALLSTIMULI/        (images, MIT1003-style) OR images/
        ALLFIXATIONMAPS/   (saliency maps with *_fixMap.jpg, MIT1003-style) OR maps/
        maps/              (saliency maps for GT overlay, standard structure)

    Expected structure for videos:
      folder_path/
        images/            (frame images)
        maps/              (saliency maps per frame for GT overlay)
    
    Examples:
      - FINAL_TEST_MIT53: images/ + maps/ (or ALLSTIMULI/ + ALLFIXATIONMAPS/)
      - SALICON: images/ + maps/ (same filename, different extensions)
      - Videos: images/ + maps/ (many frames)
    
    The function will randomly select n instances from all available images.

    Saves:
      folder_path/viz_out/{original_image_name}_viz.jpg, ...

    Run:
      # For FINAL_TEST_MIT53 (select 12 random images from 53 available):
      python run.py visualize_examples \
        --folder_path="examples/FINAL_TEST_MIT53" \
        --train_id="2026-01-09_02:04:45_unisal_debug" \
        --n=12
      
      # For SALICON (select 5 random images):
      python run.py visualize_examples \
        --folder_path="examples/salicon_test" \
        --train_id="2026-01-09_02:04:45_unisal_debug" \
        --n=5
    """
    folder_path = Path(folder_path).resolve()
    out_dir = folder_path / out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect if it's image or video dataset
    is_video = False
    img_dir = None
    maps_dir = None  # Saliency maps (GT)
    
    # Check for MIT1003-style structure (ALLSTIMULI/ALLFIXATIONMAPS)
    if (folder_path / "ALLSTIMULI").exists():
        img_dir = folder_path / "ALLSTIMULI"
        # For MIT1003, maps are in ALLFIXATIONMAPS with *_fixMap.jpg naming
        maps_dir = folder_path / "ALLFIXATIONMAPS" if (folder_path / "ALLFIXATIONMAPS").exists() else None
        is_video = False
    # Check for standard image structure (images/maps)
    elif (folder_path / "images").exists():
        img_dir = folder_path / "images"
        maps_dir = folder_path / "maps" if (folder_path / "maps").exists() else None
        # Check if it's video (has many images) or single images
        img_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg"))
        # If more than 50 images, likely a video
        if len(img_files) > 50:
            is_video = True
    else:
        raise FileNotFoundError(f"Could not find images folder. Expected 'ALLSTIMULI' or 'images' in {folder_path}")

    if not img_dir or not img_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {img_dir}")
    
    # Saliency maps are optional but recommended for GT overlay
    if maps_dir and not maps_dir.exists():
        print(f"Warning: Maps folder not found: {maps_dir}. Will skip GT overlay.")
        maps_dir = None
    elif maps_dir is None:
        print(f"Note: No saliency maps found. GT overlay will be skipped.")

    # -------- helpers --------
    def numeric_key(p: Path):
        digs = re.findall(r"\d+", p.stem)
        if not digs:
            return None
        return digs[-1].lstrip("0") or "0"

    def list_files(d: Path):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        return sorted([p for p in d.iterdir() if p.suffix.lower() in exts])

    def read_img(p: Path):
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"Could not read image: {p}")
        return im

    def read_saliency_map(p: Path):
        """Read saliency map (GT)"""
        fm = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if fm is None:
            raise ValueError(f"Could not read saliency map: {p}")
        return fm

    def normalize_0_255(gray):
        g = gray.astype(np.float32)
        g -= g.min()
        if g.max() > 1e-8:
            g /= g.max()
        return (g * 255.0).clip(0, 255).astype(np.uint8)

    def overlay_heatmap(bgr_img, heat_gray_0_255, alpha=0.55, colormap=cv2.COLORMAP_JET):
        h, w = bgr_img.shape[:2]
        heat = cv2.resize(heat_gray_0_255, (w, h), interpolation=cv2.INTER_LINEAR)
        heat_color = cv2.applyColorMap(heat, colormap)
        return cv2.addWeighted(bgr_img, 1 - alpha, heat_color, alpha, 0)

    def put_label(im, text, x, y_offset=0):
        """Put label on image at specified x position"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        color = (0, 0, 0)
        y = 30 + y_offset
        cv2.putText(im, text, (x + 12, y), font, font_scale, color, thickness, cv2.LINE_AA)

    # -------- pair images & saliency maps robustly --------
    img_files = list_files(img_dir)
    
    if is_video:
        # For videos, select n frames
        random.seed(seed)
        selected_indices = random.sample(range(len(img_files)), min(n, len(img_files)))
        pairs = [(img_files[i], None) for i in selected_indices]
        # Try to find corresponding saliency maps
        if maps_dir and maps_dir.exists():
            map_files = list_files(maps_dir)
            map_by_stem = {p.stem: p for p in map_files}
            map_by_name = {p.name: p for p in map_files}
            for i, (im, _) in enumerate(pairs):
                map_path = None
                # Try exact stem match
                if im.stem in map_by_stem:
                    map_path = map_by_stem[im.stem]
                # Try exact name match
                elif im.name in map_by_name:
                    map_path = map_by_name[im.name]
                # Try MIT1003-style: stem_fixMap.jpg
                elif (maps_dir / f"{im.stem}_fixMap.jpg").exists():
                    map_path = maps_dir / f"{im.stem}_fixMap.jpg"
                elif (maps_dir / f"{im.stem}_fixMap.png").exists():
                    map_path = maps_dir / f"{im.stem}_fixMap.png"
                if map_path:
                    pairs[i] = (im, map_path)
    else:
        # For images, pair with saliency maps
        pairs = []
        if maps_dir and maps_dir.exists():
            map_files = list_files(maps_dir)
            map_by_stem = {p.stem: p for p in map_files}
            map_by_name = {p.name: p for p in map_files}
            map_by_num = {}
            for mp in map_files:
                k = numeric_key(mp)
                if k is not None and k not in map_by_num:
                    map_by_num[k] = mp

            for im in img_files:
                map_path = None
                # Try exact stem match first
                if im.stem in map_by_stem:
                    map_path = map_by_stem[im.stem]
                # Try MIT1003-style: stem_fixMap.jpg (saliency maps)
                elif (maps_dir / f"{im.stem}_fixMap.jpg").exists():
                    map_path = maps_dir / f"{im.stem}_fixMap.jpg"
                elif (maps_dir / f"{im.stem}_fixMap.png").exists():
                    map_path = maps_dir / f"{im.stem}_fixMap.png"
                # Try exact name match
                elif im.name in map_by_name:
                    map_path = map_by_name[im.name]
                # Try numeric key matching
                else:
                    k = numeric_key(im)
                    if k is not None and k in map_by_num:
                        map_path = map_by_num[k]
                pairs.append((im, map_path))
        else:
            # No saliency maps, just use images
            pairs = [(im, None) for im in img_files]

        if len(pairs) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}")

        random.seed(seed)
        random.shuffle(pairs)
        pairs = pairs[: min(n, len(pairs))]

    # -------- load trainer + FIX the train_dir bug --------
    train_root = Path(os.environ["TRAIN_DIR"])
    real_train_dir = train_root / train_id
    if not real_train_dir.exists():
        raise FileNotFoundError(f"Train folder not found: {real_train_dir}")

    print(f"Train ID: {train_id}")
    trainer = unisal.train.Trainer.init_from_cfg_dir(real_train_dir)

    # IMPORTANT: force prefix/suffix so trainer.train_dir == real_train_dir
    # train_id format can be:
    #   - "date_time_suffix" (e.g., "2026-01-09_02:04:45_unisal_debug")
    #   - "simple_name" (e.g., "pretrained_unisal")
    parts = train_id.split("_")
    if len(parts) >= 3:
        # Standard format: date_time_suffix
        trainer.prefix = "_".join(parts[:2])      # "2026-01-09_02:04:45"
        trainer.suffix = "_".join(parts[2:])      # "unisal_debug"
    else:
        # Simple format: use the whole thing as suffix, prefix doesn't matter for existing dirs
        # The trainer.train_dir is already set correctly by init_from_cfg_dir
        trainer.suffix = train_id
        # Set a dummy prefix - it won't affect train_dir since it's already initialized
        trainer.prefix = "unknown"

    # Load weights from the REAL folder
    # Prefer weights_best.pth if it exists, else load latest checkpoint.
    weights_best = real_train_dir / "weights_best.pth"
    if weights_best.exists():
        trainer.model.load_best_weights(real_train_dir)
        print("Loaded weights_best.pth")
    else:
        # load latest chkpnt_epoch*.pth manually
        chkpnts = sorted(real_train_dir.glob("chkpnt_epoch*.pth"))
        if not chkpnts:
            raise FileNotFoundError(f"No checkpoints found in {real_train_dir}")
        last = chkpnts[-1]
        print(f"Loading checkpoint: {last.name}")
        chkpnt = torch.load(last, map_location=trainer.device)
        trainer.model.load_state_dict(chkpnt["model_state_dict"], strict=True)
        print("Loaded checkpoint model_state_dict")

    # -------- run prediction on selected images (without touching train_dir) --------
    tmp_infer = out_dir / "_tmp_infer"
    images_for_pred = tmp_infer / "images"

    # clean previous temp
    if tmp_infer.exists():
        for p in sorted(tmp_infer.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass

    images_for_pred.mkdir(parents=True, exist_ok=True)

    # copy images into temp folder
    for (im, _) in pairs:
        img = read_img(im)
        cv2.imwrite(str(images_for_pred / im.name), img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # Determine source based on dataset type
    if is_video:
        source = "DHF1K"  # Default for videos, could be UCFSports or Hollywood
        # Try to detect from folder name
        folder_name = folder_path.name.lower()
        if "ucf" in folder_name or "sports" in folder_name:
            source = "UCFSports"
        elif "hollywood" in folder_name:
            source = "Hollywood"
    else:
        source = "SALICON"  # Default for images
        if "mit" in folder_path.name.lower():
            source = "SALICON"  # MIT datasets use SALICON model domain

    # IMPORTANT: load_weights=False because we already loaded weights above
    trainer.generate_predictions_from_path(
        tmp_infer,
        is_video=is_video,
        source=source,
        load_weights=False,
        model_domain=model_domain,
    )

    pred_dir = tmp_infer / "saliency"
    if not pred_dir.exists():
        raise RuntimeError(f"Prediction folder not created: {pred_dir}")

    # -------- Load pretrained UNISAL model and generate predictions --------
    pretrained_pred_dir = None
    pretrained_train_dir = train_root / "pretrained_unisal"
    if pretrained_train_dir.exists() and train_id != "pretrained_unisal":
        print(f"\nLoading pretrained UNISAL model from {pretrained_train_dir}")
        try:
            pretrained_trainer = unisal.train.Trainer.init_from_cfg_dir(pretrained_train_dir)
            # Load pretrained weights
            pretrained_weights = pretrained_train_dir / "weights_best.pth"
            if pretrained_weights.exists():
                pretrained_trainer.model.load_best_weights(pretrained_train_dir)
                print("Loaded pretrained UNISAL weights_best.pth")
            else:
                chkpnts = sorted(pretrained_train_dir.glob("chkpnt_epoch*.pth"))
                if chkpnts:
                    last = chkpnts[-1]
                    chkpnt = torch.load(last, map_location=pretrained_trainer.device)
                    pretrained_trainer.model.load_state_dict(chkpnt["model_state_dict"], strict=True)
                    print(f"Loaded pretrained checkpoint: {last.name}")
            
            # Generate predictions with pretrained model
            pretrained_tmp_infer = out_dir / "_tmp_infer_pretrained"
            pretrained_images = pretrained_tmp_infer / "images"
            if pretrained_tmp_infer.exists():
                import shutil
                shutil.rmtree(pretrained_tmp_infer)
            pretrained_images.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            for (im, _) in pairs:
                img = read_img(im)
                cv2.imwrite(str(pretrained_images / im.name), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            pretrained_trainer.generate_predictions_from_path(
                pretrained_tmp_infer,
                is_video=is_video,
                source=source,
                load_weights=False,
                model_domain=model_domain,
            )
            pretrained_pred_dir = pretrained_tmp_infer / "saliency"
            if pretrained_pred_dir.exists():
                print("Generated pretrained UNISAL predictions")
            else:
                print("Warning: Pretrained predictions not generated")
                pretrained_pred_dir = None
        except Exception as e:
            print(f"Warning: Could not load pretrained UNISAL model: {e}")
            pretrained_pred_dir = None

    # -------- create visualizations --------
    for idx, (im_path, map_path) in enumerate(pairs, start=1):
        img = read_img(im_path)

        # Get GT saliency map if available
        gt_map = None
        if map_path and map_path.exists():
            try:
                gt_map = normalize_0_255(read_saliency_map(map_path))
            except Exception as e:
                print(f"Warning: Could not read GT saliency map {map_path}: {e}")

        # Get prediction from current model
        pred_path = pred_dir / im_path.name
        if not pred_path.exists():
            # Try alternative extensions
            for ext in [".png", ".jpg", ".jpeg"]:
                alt = pred_dir / (im_path.stem + ext)
                if alt.exists():
                    pred_path = alt
                    break
        
        if pred_path.exists():
            pred_gray = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
            if pred_gray is None:
                pred_gray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        else:
            pred_gray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        pred_gray = normalize_0_255(pred_gray)
        
        # Get pretrained prediction if available
        pretrained_pred_gray = None
        if pretrained_pred_dir and pretrained_pred_dir.exists():
            pretrained_pred_path = pretrained_pred_dir / im_path.name
            if not pretrained_pred_path.exists():
                for ext in [".png", ".jpg", ".jpeg"]:
                    alt = pretrained_pred_dir / (im_path.stem + ext)
                    if alt.exists():
                        pretrained_pred_path = alt
                        break
            
            if pretrained_pred_path.exists():
                pretrained_pred_gray = cv2.imread(str(pretrained_pred_path), cv2.IMREAD_GRAYSCALE)
                if pretrained_pred_gray is not None:
                    pretrained_pred_gray = normalize_0_255(pretrained_pred_gray)
        
        # Create panels:
        # 1. Original image
        panel1 = img.copy()
        
        # 2. Image with GT saliency map overlay
        if gt_map is not None:
            panel2 = overlay_heatmap(img, gt_map, alpha=alpha, colormap=colormap)
        else:
            panel2 = img.copy()  # No GT available
        
        # 3. Current model prediction with GT overlay
        pred_overlay = overlay_heatmap(img, pred_gray, alpha=alpha, colormap=colormap)
        if gt_map is not None:
            panel3 = overlay_heatmap(pred_overlay, gt_map, alpha=alpha*0.6, colormap=colormap)
        else:
            panel3 = pred_overlay  # Just prediction if no GT

        # 4. Pretrained UNISAL prediction with GT overlay (if available)
        if pretrained_pred_gray is not None:
            pretrained_pred_overlay = overlay_heatmap(img, pretrained_pred_gray, alpha=alpha, colormap=colormap)
            if gt_map is not None:
                panel4 = overlay_heatmap(pretrained_pred_overlay, gt_map, alpha=alpha*0.6, colormap=colormap)
            else:
                panel4 = pretrained_pred_overlay
            n_panels = 4
        else:
            panel4 = None
            n_panels = 3

        h, w = img.shape[:2]
        pad = 12
        label_h = 50 if show_labels else 0
        canvas = np.full((h + label_h, w * n_panels + pad * (n_panels - 1), 3), 255, dtype=np.uint8)
        canvas[label_h:h+label_h, 0:w] = panel1
        canvas[label_h:h+label_h, w + pad : w + pad + w] = panel2
        canvas[label_h:h+label_h, 2 * w + 2 * pad : 2 * w + 2 * pad + w] = panel3
        if n_panels == 4:
            canvas[label_h:h+label_h, 3 * w + 3 * pad : 3 * w + 3 * pad + w] = panel4

        # Add labels (optional)
        if show_labels:
            put_label(canvas, "Original", 0, 0)
            if gt_map is not None:
                put_label(canvas, "GT Map", w + pad, 0)
                put_label(canvas, f"Pred + GT", 2 * w + 2 * pad, 0)
                if n_panels == 4:
                    put_label(canvas, "Pretrained + GT", 3 * w + 3 * pad, 0)
            else:
                put_label(canvas, "No GT", w + pad, 0)
                put_label(canvas, "Prediction", 2 * w + 2 * pad, 0)
                if n_panels == 4:
                    put_label(canvas, "Pretrained", 3 * w + 3 * pad, 0)

        # Use original image name (with _viz suffix to distinguish from original)
        original_name = im_path.stem
        out_file = out_dir / f"{original_name}_viz.jpg"
        cv2.imwrite(str(out_file), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved: {out_file}")

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    fire.Fire()
