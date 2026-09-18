#!/bin/bash
# SciRS2 POLICY Refactor Tool
# Systematically converts non-core crates to use SciRS2-Core abstractions

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    cat << EOF
SciRS2 POLICY Refactor Tool

Usage: $0 [OPTIONS] CRATE_PATH

OPTIONS:
    --dry-run       Show what would be changed without making changes
    --backup        Create backup before making changes
    --verify        Run cargo check after refactoring
    --help          Show this help message

EXAMPLES:
    $0 scirs2-stats                    # Refactor scirs2-stats crate
    $0 --dry-run scirs2-linalg         # Preview changes for scirs2-linalg
    $0 --backup --verify scirs2-neural # Backup, refactor, and verify

SUPPORTED REFACTORS:
    - rand:: -> scirs2_core::random::
    - ndarray:: -> scirs2_core::array::
    - Common API patterns and types
EOF
}

# Function to find all Rust files excluding target/
find_rust_files() {
    local search_path="$1"
    find "$search_path" -name "*.rs" -not -path "*/target/*" -type f
}

# Function to backup files
backup_files() {
    local crate_path="$1"
    local backup_dir="$crate_path.backup.$(date +%s)"

    log_info "Creating backup at $backup_dir"
    cp -r "$crate_path" "$backup_dir"
    echo "$backup_dir"
}

# Function to refactor imports
refactor_imports() {
    local file="$1"
    local dry_run="$2"

    # Check if file needs refactoring
    local needs_refactor=false
    if grep -q "^use rand::" "$file" 2>/dev/null; then
        needs_refactor=true
    fi
    if grep -q "^use ndarray::" "$file" 2>/dev/null; then
        needs_refactor=true
    fi
    if grep -q "rand::" "$file" 2>/dev/null; then
        needs_refactor=true
    fi

    if [ "$needs_refactor" = "false" ]; then
        return
    fi

    if [ "$dry_run" = "true" ]; then
        echo "  📝 Would refactor: $(basename "$file")"
        return
    fi

    log_info "Refactoring: $(basename "$file")"

    # Phase 1: Import replacements with careful handling of mixed patterns
    # Basic import replacements
    sed -i '' 's/^use rand::/use scirs2_core::random::/g' "$file"
    sed -i '' 's/^use ndarray::/use scirs2_core::ndarray_ext::/g' "$file"

    # Handle mixed rand imports - need to preserve SliceRandom trait
    if grep -q "use rand::prelude::\*;" "$file"; then
        sed -i '' 's/use scirs2_core::random::prelude::\*;/use scirs2_core::random::*;\nuse rand::seq::SliceRandom;/g' "$file"
    fi

    if grep -q "use scirs2_core::random::seq::SliceRandom;" "$file"; then
        sed -i '' 's/use scirs2_core::random::seq::SliceRandom;/use rand::seq::SliceRandom;/g' "$file"
    fi

    # Fix StdRng imports - scirs2-core re-exports but keep rand version for compatibility
    if grep -q "use scirs2_core::random::rngs::StdRng;" "$file"; then
        sed -i '' 's/use scirs2_core::random::rngs::StdRng;/use scirs2_core::random::Random;\nuse rand::rngs::StdRng;\nuse rand::SeedableRng;/g' "$file"
    fi

    # Phase 2: Type and function call replacements
    # Fix RNG creation patterns - convert StdRng to Random<StdRng>
    sed -i '' 's/StdRng::seed_from_u64(\([^)]*\))/Random::seed(\1)/g' "$file"

    # Fix direct rng() calls to use scirs2-core function
    sed -i '' 's/let mut r = rng();/let mut r = scirs2_core::random::thread_rng();/g' "$file"
    sed -i '' 's/\([^a-zA-Z_]\)rng()/\1scirs2_core::random::thread_rng()/g' "$file"

    # Fix .next_u64() calls on Random struct
    sed -i '' 's/\.next_u64()/.rng_mut().next_u64()/g' "$file"

    # Fix common API patterns
    sed -i '' 's/rand::thread_rng()/scirs2_core::random::thread_rng()/g' "$file"

    # Phase 3: Fix shuffle calls for Random struct
    # Convert slice.shuffle(&mut rng) to slice.shuffle(&mut rng.rng) for Random<T> types
    sed -i '' 's/\.shuffle(&mut \([a-zA-Z_][a-zA-Z0-9_]*\));/.shuffle(\&mut \1.rng);/g' "$file"

    # Phase 4: Add missing trait imports if shuffle is used
    if grep -q "\.shuffle(" "$file" && ! grep -q "use rand::seq::SliceRandom;" "$file"; then
        # Add SliceRandom import after existing imports
        sed -i '' '/^use scirs2_core::random/a\
use rand::seq::SliceRandom;
' "$file"
    fi

    # Phase 5: Clean up any double imports that might have been created
    sed -i '' '/^use rand::seq::SliceRandom;$/N; /\n.*use rand::seq::SliceRandom;$/d' "$file"
}

# Function to verify compilation
verify_compilation() {
    local crate_path="$1"

    log_info "Verifying compilation for $crate_path"
    cd "$crate_path"

    if cargo check --lib --all-features 2>/dev/null; then
        log_info "✅ Compilation successful"
        return 0
    else
        log_error "❌ Compilation failed"
        return 1
    fi
}

# Function to analyze refactor scope
analyze_scope() {
    local crate_path="$1"

    log_info "Analyzing refactor scope for $crate_path"

    local rand_count=$(find_rust_files "$crate_path" | xargs grep -l "^use rand::" | wc -l | tr -d ' ')
    local ndarray_count=$(find_rust_files "$crate_path" | xargs grep -l "^use ndarray::" | wc -l | tr -d ' ')

    echo "Files needing refactor:"
    echo "  - rand:: imports: $rand_count files"
    echo "  - ndarray:: imports: $ndarray_count files"
    echo "  - Total files: $((rand_count + ndarray_count)) files"
}

# Main refactor function
refactor_crate() {
    local crate_path="$1"
    local dry_run="$2"
    local create_backup="$3"
    local verify="$4"

    if [ ! -d "$crate_path" ]; then
        log_error "Crate path does not exist: $crate_path"
        return 1
    fi

    log_info "Starting SciRS2 POLICY refactor for: $crate_path"

    # Analyze scope first
    analyze_scope "$crate_path"

    # Create backup if requested
    local backup_dir=""
    if [ "$create_backup" = "true" ] && [ "$dry_run" != "true" ]; then
        backup_dir=$(backup_files "$crate_path")
    fi

    # Refactor all Rust files
    local file_count=0
    local refactored_count=0

    echo "Files that need refactoring:"
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            ((file_count++))
            # Check if file needs refactoring before calling the function
            if grep -q "^use rand::\|^use ndarray::" "$file" 2>/dev/null; then
                refactor_imports "$file" "$dry_run"
                ((refactored_count++))
            fi
        fi
    done < <(find_rust_files "$crate_path")

    if [ "$dry_run" = "true" ]; then
        log_info "DRY RUN: Would refactor $refactored_count out of $file_count total files"
        return 0
    fi

    log_info "Refactored $refactored_count out of $file_count total files"

    # Verify compilation if requested
    if [ "$verify" = "true" ]; then
        if ! verify_compilation "$crate_path"; then
            if [ -n "$backup_dir" ]; then
                log_warn "Restoring backup due to compilation failure"
                rm -rf "$crate_path"
                mv "$backup_dir" "$crate_path"
            fi
            return 1
        fi
    fi

    log_info "✅ Refactor complete for $crate_path"
    return 0
}

# Parse command line arguments
DRY_RUN=false
CREATE_BACKUP=false
VERIFY=false
CRATE_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --backup)
            CREATE_BACKUP=true
            shift
            ;;
        --verify)
            VERIFY=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            if [ -z "$CRATE_PATH" ]; then
                CRATE_PATH="$1"
            else
                log_error "Multiple crate paths specified"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate arguments
if [ -z "$CRATE_PATH" ]; then
    log_error "No crate path specified"
    show_usage
    exit 1
fi

# Convert relative path to absolute
if [[ ! "$CRATE_PATH" = /* ]]; then
    CRATE_PATH="$WORKSPACE_ROOT/$CRATE_PATH"
fi

# Run the refactor
refactor_crate "$CRATE_PATH" "$DRY_RUN" "$CREATE_BACKUP" "$VERIFY"