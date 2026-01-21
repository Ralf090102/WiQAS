#!/bin/bash

# Knowledge Base Transfer Script for GCP
# Helps transfer the 5GB knowledge base to GCP VM

set -e

echo "=========================================="
echo "   WiQAS Knowledge Base Transfer"
echo "=========================================="
echo ""

# Configuration - Update these based on your target VM
LOCAL_KB_PATH="./data/knowledge_base"
REMOTE_KB_PATH="~/WiQAS/data/knowledge_base"
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
echo "  1) Google Cloud Storage (Recommended for 5GB)"
echo "  2) Direct SCP via gcloud (Slower but simpler)"
echo "  3) rsync (Best for incremental updates)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        print_info "Using Google Cloud Storage..."
        
        # Get or create bucket name
        read -p "Enter GCS bucket name (or press Enter for default 'wiqas-kb'): " input_bucket
        BUCKET_NAME=${input_bucket:-"wiqas-kb-$(date +%s)"}
        
        prini in "${!TARGET_VMS[@]}"; do
            VM_NAME="${TARGET_VMS[$i]}"
            ZONE="${TARGET_ZONES[$i]}"
            print_info "Downloading to VM: $VM_NAME (zone: $ZONE)NAME" 2>/dev/null || print_info "Bucket already exists"
        
        print_info "Uploading knowledge base to GCS (5GB - this may take several minutes)..."
        gsutil -m cp -r "$LOCAL_KB_PATH"/* "gs://$BUCKET_NAME/"
        
        print_success "Upload to GCS complete!"
        
        # Download to each VM
        for VM_NAME in "${TARGET_VMS[@]}"; do
            print_info "Downloading to VM: $VM_NAME"
            gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
                mkdir -p ~/WiQAS/data/knowledge_base && 
                echo 'Downloading from GCS...' &&
                gsutil -m cp -r gs://$BUCKET_NAME/* ~/WiQAS/data/knowledge_base/ &&
                echo '✅ Knowledge base downloaded successfully'
            "
            print_success "Transfer complete to $VM_NAME"
        done
        ;;
        
    2)
        print_info "Using direct SCP transfer..."
        
        # Compress first
        print_info "Compressing knowledge base (5GB - this may take a few minutes)..."
        tar -czf knowledge_base.tar.gz -C ./data knowledge_base
        
        KB_SIZE=$(du -h knowledge_base.tar.gz | cut -f1)
        print_success "Compressed to: $KB_SIZE"
        i in "${!TARGET_VMS[@]}"; do
            VM_NAME="${TARGET_VMS[$i]}"
            ZONE="${TARGET_ZONES[$i]}"
            print_info "Transferring to VM: $VM_NAME (zone: $ZONE)
        for VM_NAME in "${TARGET_VMS[@]}"; do
            print_info "Transferring to VM: $VM_NAME"
            
            gcloud compute scp knowledge_base.tar.gz "$VM_NAME:~/WiQAS/data/" --zone="$ZONE"
            
            print_info "Extracting on VM..."
            gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
                cd ~/WiQAS/data && 
                tar -xzf knowledge_base.tar.gz && 
                rm knowledge_base.tar.gz &&
                echo '✅ Knowledge base extracted successfully'
            "
            
            print_success "Transfer complete to $VM_NAME"
        done
        
        # Clean up local compressed file
        rm knowledge_base.tar.gz
        print_info "Cleaned up local compressed file"
        ;;
        
    3)
        # Sync to each VM
        for i in "${!TARGET_VMS[@]}"; do
            VM_NAME="${TARGET_VMS[$i]}"
            ZONE="${TARGET_ZONES[$i]}"
            print_info "Syncing to VM: $VM_NAME (zone: $ZONE)re'"
echo "
            print_info "Syncing to VM: $VM_NAME"
            
            rsync -avz --progress -e "gcloud compute ssh --zone=$ZONE" \
                "$LOCAL_KB_PATH/" \
                "$VM_NAME:$REMOTE_KB_PATH/"
            
            print_success "Sync complete to $VM_NAME"
        done
        ;;
        
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
print_success "Knowledge base transfer completed!"
print_info "Next step: Ingest documents on VM with:"
echo "  python run.py ingest ./data/knowledge_base/ --workers 8"
=========================================="
print_success "Knowledge base transfer completed!"
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
echo "  3. Verify knowledge base:"
echo "     ls -lh data/knowledge_base/"
echo "     du -sh data/knowledge_base/"
echo ""
echo "  4. Ingest documents:"
echo "     python run.py ingest ./data/knowledge_base/ --workers 8"
echo ""
echo "  5. Test:"
echo "     python run.py status"
echo "     python run.py search 'Filipino culture'"
echo "