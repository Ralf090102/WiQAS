#!/bin/bash

# Chroma Vector Database Transfer Script for GCP
# Helps transfer the embedded chroma-data to GCP VM

set -e

echo "=========================================="
echo "   WiQAS Vector Database Transfer"
echo "=========================================="
echo ""

# Configuration - Update these based on your target VM
LOCAL_CHROMA_PATH="./data/chroma-data"
REMOTE_CHROMA_PATH="~/WiQAS/data/chroma-data"
BUCKET_NAME=""

# VM Names and Zones
VM_EMBEDDING="wiqas-embedding-reranking-20260120-045555"
ZONE_EMBEDDING="asia-southeast1-b"
VM_GENERATION="wiqas-generation-evaluation-20260120-061404"
ZONE_GENERATION="asia-southeast1-c"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Select target VM
echo "Select target VM:"
echo "  1) Embedding + Reranking VM (T4) - $VM_EMBEDDING"
echo "  2) Generation + Evaluation VM (A100) - $VM_GENERATION"
echo "  3) Both VMs"
echo ""
read -p "Enter choice [1-3]: " vm_choice

case $vm_choice in
    1) 
        TARGET_VMS=("$VM_EMBEDDING")
        TARGET_ZONES=("$ZONE_EMBEDDING")
        ;;
    2) 
        TARGET_VMS=("$VM_GENERATION")
        TARGET_ZONES=("$ZONE_GENERATION")
        ;;
    3) 
        TARGET_VMS=("$VM_EMBEDDING" "$VM_GENERATION")
        TARGET_ZONES=("$ZONE_EMBEDDING" "$ZONE_GENERATION")
        ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

echo ""
echo "Choose transfer method:"
echo "  1) Google Cloud Storage (Recommended for large chroma-data)"
echo "  2) Direct SCP via gcloud (Slower but simpler)"
echo "  3) rsync (Best for incremental updates)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        print_info "Using Google Cloud Storage..."
        
        # Get or create bucket name
        read -p "Enter GCS bucket name (or press Enter for default 'wiqas-chroma'): " input_bucket
        BUCKET_NAME=${input_bucket:-"wiqas-chroma-$(date +%s)"}
        
        print_info "Creating bucket in asia-southeast1: gs://$BUCKET_NAME"
        gsutil mb -l asia-southeast1 "gs://$BUCKET_NAME" 2>/dev/null || print_info "Bucket already exists"
        
        print_info "Uploading chroma-data to GCS (this may take several minutes)..."
        # Use -r to recursively copy the entire directory including subdirectories
        gsutil -m rsync -r "$LOCAL_CHROMA_PATH" "gs://$BUCKET_NAME/"
        
        print_success "Upload to GCS complete!"
        
        # Download to each VM
        for i in "${!TARGET_VMS[@]}"; do
            VM_NAME="${TARGET_VMS[$i]}"
            ZONE="${TARGET_ZONES[$i]}"
            print_info "Downloading to VM: $VM_NAME (zone: $ZONE)"
            gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
                mkdir -p ~/WiQAS/data/chroma-data && 
                echo 'Downloading from GCS...' &&
                gsutil -m rsync -r gs://$BUCKET_NAME/ ~/WiQAS/data/chroma-data/ &&
                echo '✅ Chroma database downloaded successfully'
            "
            print_success "Transfer complete to $VM_NAME"
        done
        ;;
        
    2)
        print_info "Using direct SCP transfer..."
        
        # Compress first
        print_info "Compressing chroma-data (this may take a few minutes)..."
        tar -czf chroma-data.tar.gz -C ./data chroma-data
        
        CHROMA_SIZE=$(du -h chroma-data.tar.gz | cut -f1)
        print_success "Compressed to: $CHROMA_SIZE"
        
        # Transfer to each VM
        for i in "${!TARGET_VMS[@]}"; do
            VM_NAME="${TARGET_VMS[$i]}"
            ZONE="${TARGET_ZONES[$i]}"
            print_info "Transferring to VM: $VM_NAME (zone: $ZONE)"
            
            gcloud compute scp chroma-data.tar.gz "$VM_NAME:/WiQAS/data/" --zone="$ZONE"
            
            print_info "Extracting on VM..."
            gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
                cd ~/WiQAS/data && 
                tar -xzf chroma-data.tar.gz && 
                rm chroma-data.tar.gz &&
                echo '✅ Chroma database extracted successfully'
            "
            
            print_success "Transfer complete to $VM_NAME"
        done
        
        # Clean up local compressed file
        rm chroma-data.tar.gz
        print_info "Cleaned up local compressed file"
        ;;
        
    3)
        print_info "Using rsync (incremental transfer)..."
        
        # Sync to each VM
        for i in "${!TARGET_VMS[@]}"; do
            VM_NAME="${TARGET_VMS[$i]}"
            ZONE="${TARGET_ZONES[$i]}"
            print_info "Syncing to VM: $VM_NAME (zone: $ZONE)"
            
            rsync -avz --progress -e "gcloud compute ssh --zone=$ZONE" \
                "$LOCAL_CHROMA_PATH/" \
                "$VM_NAME:$REMOTE_CHROMA_PATH/"
            
            print_success "Sync complete to $VM_NAME"
        done
        ;;
        
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
print_success "=========================================="
print_success "Chroma database transfer completed!"
print_success "=========================================="
echo ""
print_info "Next steps on each VM:"
echo ""
echo "  1. SSH into VM:"
echo "     Embedding VM: gcloud compute ssh $VM_EMBEDDING --zone=$ZONE_EMBEDDING"
echo "     Generation VM: gcloud compute ssh $VM_GENERATION --zone=$ZONE_GENERATION"
echo ""
echo "  2. Activate environment:"
echo "     source ~/wiqas-venv/bin/activate"
echo "     cd ~/WiQAS"
echo ""
echo "  3. Verify chroma database:"
echo "     ls -lh data/chroma-data/"
echo "     du -sh data/chroma-data/"
echo ""
echo "  4. Test the system (no ingestion needed!):"
echo "     python run.py status"
echo "     python run.py search 'Filipino culture'"
echo "     python run.py ask 'What is bayanihan?'"
echo ""