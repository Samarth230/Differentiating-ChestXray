#!/usr/bin/env python3
"""
Main script for Medical Image Classification Project
Provides a unified interface for training, evaluation, and prediction
"""

import os
import sys
import argparse
import json
from pathlib import Path

from config import get_config, validate_config, print_config_summary
from data_preprocessing import DataPreprocessor
from model import create_model, get_model_summary
from train import MedicalImageTrainer
from evaluate import ModelEvaluator
from predict import MedicalImagePredictor

def setup_environment():
    """Setup the environment and check dependencies"""
    print("Setting up environment...")
    
    required_dirs = ['data', 'outputs', 'evaluation_results', 'prediction_results']
    for dir_name in required_dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"  ✓ {dir_name}/ directory ready")
    
    data_dir = Path('data')
    if data_dir.exists():
        subdirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
        expected_classes = ['normal', 'pneumonia', 'tuberculosis']
        
        print(f"  Data directory contains: {subdirs}")
        if any(cls in subdirs for cls in expected_classes):
            print("  ✓ Data structure appears correct")
        else:
            print("  ⚠️  Expected class directories not found")
            print("     Please ensure your data directory contains:")
            print("     data/")
            print("     ├── normal/")
            print("     ├── pneumonia/")
            print("     └── tuberculosis/")
    else:
        print("  ⚠️  Data directory not found")
        print("     Please create the data directory with the expected structure")
    
    print("Environment setup completed!\n")

def check_data_availability(data_dir: str = "data"):
    """Check if data is available for training"""
    if not os.path.exists(data_dir):
        return False, "Data directory does not exist"
    
    class_counts = {}
    total_images = 0
    
    for class_name in ['normal', 'pneumonia', 'tuberculosis']:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
            class_counts[class_name] = len(images)
            total_images += len(images)
        else:
            class_counts[class_name] = 0
    
    if total_images == 0:
        return False, "No images found in data directory"
    
    min_images_per_class = 100
    for class_name, count in class_counts.items():
        if count < min_images_per_class:
            return False, f"Insufficient images in {class_name} class ({count} < {min_images_per_class})"
    
    return True, f"Data available: {total_images} total images across {len(class_counts)} classes"

def train_model(config: dict, data_dir: str = "data"):
    """Train the medical image classification model"""
    print("=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)
    
    data_available, message = check_data_availability(data_dir)
    if not data_available:
        print(f"❌ {message}")
        print("Please prepare your dataset before training.")
        return False
    
    print(f"✅ {message}")
    
    try:
        trainer = MedicalImageTrainer(config)
        trainer.prepare_data(data_dir)
        trainer.train()
        
        print("✅ Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def evaluate_model(config: dict, model_path: str = "outputs/best_model.pth"):
    """Evaluate the trained model"""
    print("=" * 60)
    print("EVALUATION PHASE")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train the model first or specify the correct model path.")
        return False
    
    try:
        evaluator = ModelEvaluator(config, model_path)
        data_dir = "data"
        evaluator.prepare_test_data(data_dir)
        evaluator.evaluate_model()
        
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        evaluator.create_visualizations(output_dir)
        evaluator.generate_detailed_report(output_dir)
        evaluator.analyze_predictions(output_dir)
        
        print("✅ Evaluation completed successfully!")
        print(f"Results saved to {output_dir}/")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

def predict_images(config: dict, model_path: str = "outputs/best_model.pth", 
                  image_paths: list = None):
    """Make predictions on new images"""
    print("=" * 60)
    print("PREDICTION PHASE")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train the model first or specify the correct model path.")
        return False
    
    if not image_paths:
        print("❌ No image paths provided for prediction.")
        print("Please specify image paths using --images argument.")
        return False
    
    try:
        predictor = MedicalImagePredictor(config, model_path)
        
        all_image_files = []
        for path in image_paths:
            if os.path.isfile(path):
                all_image_files.append(path)
            elif os.path.isdir(path):
                print(f"📁 Scanning directory: {path}")
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            full_path = os.path.join(root, file)
                            all_image_files.append(full_path)
                print(f"   Found {len([f for f in all_image_files if f.startswith(path)])} images in {path}")
            else:
                print(f"⚠️  Path not found: {path}")
        
        if not all_image_files:
            print("❌ No valid image files found.")
            return False
        
        print(f"🎯 Total images to process: {len(all_image_files)}")
        
        results = []
        for i, image_path in enumerate(all_image_files, 1):
            try:
                print(f"Processing [{i}/{len(all_image_files)}]: {os.path.basename(image_path)}")
                result = predictor.predict_single_image(image_path, return_attention=False)
                result['image_path'] = image_path
                results.append(result)
                
                print(f"  Predicted: {result['predicted_class']}")
                print(f"  Confidence: {result['confidence']:.3f}")
            except Exception as e:
                print(f"  ❌ Error processing {os.path.basename(image_path)}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'predicted_class': 'Error',
                    'confidence': 0.0
                })
        
        if results:
            output_dir = "prediction_results"
            os.makedirs(output_dir, exist_ok=True)
            
            for result in results:
                if 'error' not in result:
                    image_path = result['image_path']
                    viz_path = os.path.join(output_dir, f"prediction_{os.path.basename(image_path)}.png")
                    predictor.visualize_prediction(image_path, result, viz_path)
            
            report_path = os.path.join(output_dir, 'prediction_report.csv')
            predictor.create_prediction_report(results, report_path)
            predictor.analyze_prediction_confidence(results, output_dir)
            
            print("✅ Prediction completed successfully!")
            print(f"Results saved to {output_dir}/")
            return True
        else:
            print("❌ No valid images processed.")
            return False
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

def show_model_info(config: dict):
    """Display model architecture information"""
    print("=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    
    try:
        model = create_model(config['model'])
        print(get_model_summary(model))
        print_config_summary(config)
        
    except Exception as e:
        print(f"❌ Failed to display model information: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Medical Image Classification: Normal, Pneumonia, and Tuberculosis Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python main.py --train
  
  # Evaluate the trained model
  python main.py --evaluate
  
  # Make predictions on images
  python main.py --predict --images image1.jpg image2.jpg
  
  # Show model information
  python main.py --info
  
  # Full pipeline (train + evaluate)
  python main.py --full-pipeline
  
  # Custom configuration
  python main.py --train --config custom_config.json
        """
    )
    
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the trained model')
    parser.add_argument('--predict', action='store_true', help='Make predictions on new images')
    parser.add_argument('--info', action='store_true', help='Show model information')
    parser.add_argument('--full-pipeline', action='store_true', help='Run full pipeline (train + evaluate)')
    
    parser.add_argument('--config', type=str, default='complete', 
                       help='Configuration to use (default: complete)')
    parser.add_argument('--config-file', type=str, help='Path to custom configuration file')
    
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory path')
    parser.add_argument('--model-path', type=str, default='outputs/best_model.pth', 
                       help='Path to trained model')
    
    parser.add_argument('--images', nargs='+', help='Image paths for prediction')
    
    args = parser.parse_args()
    
    if not any([args.train, args.evaluate, args.predict, args.info, args.full_pipeline]):
        parser.print_help()
        return
    
    print("🏥 Medical Image Classification System")
    print("=" * 60)
    
    setup_environment()
    
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded custom configuration from {args.config_file}")
    else:
        config = get_config(args.config)
        print(f"✅ Using built-in configuration: {args.config}")
    
    if not validate_config(config):
        print("❌ Configuration validation failed!")
        return
    
    if args.info:
        show_model_info(config)
        return
    
    success = True
    
    if args.train or args.full_pipeline:
        print("\n🚀 Starting training phase...")
        success &= train_model(config, args.data_dir)
    
    if args.evaluate or args.full_pipeline:
        print("\n📊 Starting evaluation phase...")
        success &= evaluate_model(config, args.model_path)
    
    if args.predict:
        print("\n🔮 Starting prediction phase...")
        success &= predict_images(config, args.model_path, args.images)
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All operations completed successfully!")
        print("\nNext steps:")
        if args.train or args.full_pipeline:
            print("  • Check training results in outputs/ directory")
            print("  • View TensorBoard logs: tensorboard --logdir outputs/logs")
        if args.evaluate or args.full_pipeline:
            print("  • Review evaluation results in evaluation_results/ directory")
        if args.predict:
            print("  • Check prediction results in prediction_results/ directory")
    else:
        print("❌ Some operations failed. Please check the error messages above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
