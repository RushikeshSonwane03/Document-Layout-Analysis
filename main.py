import requests
import json
import os
import time
from PIL import Image
import io
import base64

def download_doclaynet_via_api():
    """Download DocLayNet data using the datasets-server API with pagination"""
    
    print("🚀 DocLayNet Download via Datasets-Server API")
    print("This will download sample data using the working API endpoint")
    print()
    
    # Create data directory
    data_dir = "./data/DocLayNet_API"
    os.makedirs(data_dir, exist_ok=True)
    
    # API configuration
    base_url = "https://datasets-server.huggingface.co/rows"
    dataset_name = "ds4sd/DocLayNet"
    config = "2022.08"
    
    # Download data for each split
    splits = ["train", "validation", "test"]
    
    all_data = {}
    
    for split in splits:
        print(f"\n📥 Downloading {split} split..." )
        
        split_dir = os.path.join(data_dir, split)
        images_dir = os.path.join(split_dir, "images")
        annotations_dir = os.path.join(split_dir, "annotations")
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)
        
        split_data = []
        offset = 0
        batch_size = 100
        max_samples = 1000  # Adjust this to download more samples
        
        while len(split_data) < max_samples:
            print(f"  Fetching batch at offset {offset}...")
            
            # Make API request
            params = {
                "dataset": dataset_name,
                "config": config,
                "split": split,
                "offset": offset,
                "length": batch_size
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if "rows" not in data or not data["rows"]:
                    print(f"  No more data available at offset {offset}")
                    break
                
                batch_rows = data["rows"]
                print(f"  Retrieved {len(batch_rows)} samples")
                
                # Process each row
                for row_data in batch_rows:
                    row = row_data["row"]
                    
                    # Extract image data
                    if "image" in row and row["image"]:
                        image_info = row["image"]
                        
                        # Save image
                        if "bytes" in image_info and image_info["bytes"]:
                            try:
                                # Decode base64 image
                                image_bytes = base64.b64decode(image_info["bytes"])
                                image = Image.open(io.BytesIO(image_bytes))
                                
                                # Create filename
                                image_id = row.get("image_id", f"{split}_{len(split_data)}")
                                image_filename = f"{image_id}.png"
                                image_path = os.path.join(images_dir, image_filename)
                                
                                # Save image
                                image.save(image_path)
                                
                                # Create annotation data
                                annotation = {
                                    "image_id": image_id,
                                    "width": row.get("width", image.width),
                                    "height": row.get("height", image.height),
                                    "doc_category": row.get("doc_category", ""),
                                    "collection": row.get("collection", ""),
                                    "doc_name": row.get("doc_name", ""),
                                    "page_no": row.get("page_no", 0),
                                    "objects": row.get("objects", {})
                                }
                                
                                # Save annotation
                                ann_filename = f"{image_id}.json"
                                ann_path = os.path.join(annotations_dir, ann_filename)
                                with open(ann_path, 'w') as f:
                                    json.dump(annotation, f, indent=2)
                                
                                split_data.append(annotation)
                                
                                if len(split_data) % 50 == 0:
                                    print(f"    Processed {len(split_data)} samples...")
                                
                            except Exception as e:
                                print(f"    Error processing image: {e}")
                                continue
                
                offset += batch_size
                
                # Small delay to be respectful to the API
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                print(f"  API request failed: {e}")
                break
            except Exception as e:
                print(f"  Error processing batch: {e}")
                break
        
        all_data[split] = split_data
        
        # Save combined annotations
        combined_path = os.path.join(split_dir, f"{split}_annotations.json")
        with open(combined_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"✅ {split}: {len(split_data)} samples downloaded")
    
    # Create summary
    total_samples = sum(len(data) for data in all_data.values())
    
    summary = {
        "dataset_name": "DocLayNet (API Sample)",
        "download_method": "datasets-server API with pagination",
        "splits": {split: len(data) for split, data in all_data.items()},
        "total_samples": total_samples,
        "dataset_path": data_dir,
        "note": "This is a sample of the full dataset obtained via API"
    }
    
    with open(os.path.join(data_dir, "download_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Download Summary:")
    for split, count in summary["splits"].items():
        print(f"  {split}: {count} samples")
    print(f"  Total: {total_samples} samples")
    print(f"📁 Data saved to: {os.path.abspath(data_dir)}")
    
    if total_samples > 100:
        print("\n🎉 SUCCESS! Downloaded substantial DocLayNet sample data!")
        print("You now have:")
        print("- Real DocLayNet images and annotations")
        print("- Proper dataset structure")
        print("- Enough data for development and testing")
        return True
    else:
        print("\n⚠️  Downloaded limited data. API may have restrictions.")
        return False

if __name__ == "__main__":
    print("DocLayNet API Downloader")
    print("=" * 30)
    
    response = input("\nDownload DocLayNet sample via API? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Download cancelled.")
        exit(0)
    
    success = download_doclaynet_via_api()
    
    if success:
        print("\n🎉 You now have real DocLayNet data to work with!")
        print("This should be sufficient for developing your Flask application.")
    else:
        print("\n❌ API download had issues, but you may still have some data.")
