-- Create enums
CREATE TYPE user_role AS ENUM ('admin', 'editor', 'viewer', 'guest');
CREATE TYPE permission_type AS ENUM ('read', 'write', 'delete', 'share', 'admin');

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role user_role NOT NULL DEFAULT 'viewer',
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);

-- Assets table
CREATE TABLE IF NOT EXISTS assets (
    id UUID PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(1024) NOT NULL,
    file_size BIGINT,
    mime_type VARCHAR(255),
    checksum VARCHAR(64) NOT NULL,

    -- Technical metadata
    duration_ms BIGINT,
    width INTEGER,
    height INTEGER,
    frame_rate DECIMAL(10, 4),
    video_codec VARCHAR(255),
    audio_codec VARCHAR(255),
    bit_rate BIGINT,

    -- Descriptive metadata
    title VARCHAR(512),
    description TEXT,
    keywords TEXT[],
    categories TEXT[],

    -- Rights metadata
    copyright VARCHAR(512),
    license VARCHAR(255),
    creator VARCHAR(255),

    -- Custom metadata (JSONB for flexibility)
    custom_metadata JSONB,

    -- Status and tracking
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_checksum UNIQUE(checksum)
);

CREATE INDEX idx_assets_checksum ON assets(checksum);
CREATE INDEX idx_assets_filename ON assets(filename);
CREATE INDEX idx_assets_mime_type ON assets(mime_type);
CREATE INDEX idx_assets_created_at ON assets(created_at DESC);
CREATE INDEX idx_assets_status ON assets(status);
CREATE INDEX idx_assets_keywords ON assets USING GIN(keywords);
CREATE INDEX idx_assets_custom_metadata ON assets USING GIN(custom_metadata);

-- Collections table
CREATE TABLE IF NOT EXISTS collections (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    parent_id UUID REFERENCES collections(id) ON DELETE CASCADE,
    is_smart BOOLEAN NOT NULL DEFAULT false,
    smart_query JSONB,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_collections_parent ON collections(parent_id);
CREATE INDEX idx_collections_name ON collections(name);
CREATE INDEX idx_collections_created_at ON collections(created_at DESC);

-- Collection items (many-to-many)
CREATE TABLE IF NOT EXISTS collection_items (
    id UUID PRIMARY KEY,
    collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    position INTEGER,
    added_by UUID REFERENCES users(id),
    added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_collection_asset UNIQUE(collection_id, asset_id)
);

CREATE INDEX idx_collection_items_collection ON collection_items(collection_id);
CREATE INDEX idx_collection_items_asset ON collection_items(asset_id);

-- Asset versions
CREATE TABLE IF NOT EXISTS asset_versions (
    id UUID PRIMARY KEY,
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    file_path VARCHAR(1024) NOT NULL,
    file_size BIGINT NOT NULL,
    checksum VARCHAR(64) NOT NULL,
    created_by UUID NOT NULL REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    comment TEXT,
    CONSTRAINT unique_asset_version UNIQUE(asset_id, version_number)
);

CREATE INDEX idx_asset_versions_asset ON asset_versions(asset_id);

-- Asset permissions
CREATE TABLE IF NOT EXISTS asset_permissions (
    id UUID PRIMARY KEY,
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role user_role,
    permission permission_type NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT check_user_or_role CHECK (user_id IS NOT NULL OR role IS NOT NULL)
);

CREATE INDEX idx_asset_permissions_asset ON asset_permissions(asset_id);
CREATE INDEX idx_asset_permissions_user ON asset_permissions(user_id);

-- Collection permissions
CREATE TABLE IF NOT EXISTS collection_permissions (
    id UUID PRIMARY KEY,
    collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role user_role,
    permission permission_type NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT check_user_or_role_coll CHECK (user_id IS NOT NULL OR role IS NOT NULL)
);

CREATE INDEX idx_collection_permissions_collection ON collection_permissions(collection_id);
CREATE INDEX idx_collection_permissions_user ON collection_permissions(user_id);

-- Workflows
CREATE TABLE IF NOT EXISTS workflows (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    workflow_type VARCHAR(50) NOT NULL,
    config JSONB NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_workflows_type ON workflows(workflow_type);
CREATE INDEX idx_workflows_active ON workflows(is_active);

-- Workflow instances
CREATE TABLE IF NOT EXISTS workflow_instances (
    id UUID PRIMARY KEY,
    workflow_id UUID NOT NULL REFERENCES workflows(id),
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    current_state VARCHAR(255),
    state_data JSONB,
    started_by UUID REFERENCES users(id),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error_message TEXT
);

CREATE INDEX idx_workflow_instances_workflow ON workflow_instances(workflow_id);
CREATE INDEX idx_workflow_instances_asset ON workflow_instances(asset_id);
CREATE INDEX idx_workflow_instances_status ON workflow_instances(status);

-- Workflow tasks
CREATE TABLE IF NOT EXISTS workflow_tasks (
    id UUID PRIMARY KEY,
    instance_id UUID NOT NULL REFERENCES workflow_instances(id) ON DELETE CASCADE,
    task_type VARCHAR(255) NOT NULL,
    assigned_to UUID REFERENCES users(id),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    due_date TIMESTAMPTZ,
    completed_by UUID REFERENCES users(id),
    completed_at TIMESTAMPTZ,
    comment TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_workflow_tasks_instance ON workflow_tasks(instance_id);
CREATE INDEX idx_workflow_tasks_assigned ON workflow_tasks(assigned_to);
CREATE INDEX idx_workflow_tasks_status ON workflow_tasks(status);

-- Comments/annotations
CREATE TABLE IF NOT EXISTS comments (
    id UUID PRIMARY KEY,
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id),
    parent_id UUID REFERENCES comments(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    timecode_ms BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_comments_asset ON comments(asset_id);
CREATE INDEX idx_comments_user ON comments(user_id);
CREATE INDEX idx_comments_parent ON comments(parent_id);

-- Ingest jobs
CREATE TABLE IF NOT EXISTS ingest_jobs (
    id UUID PRIMARY KEY,
    source_path VARCHAR(1024) NOT NULL,
    asset_id UUID REFERENCES assets(id),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    progress INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    metadata JSONB,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_ingest_jobs_status ON ingest_jobs(status);
CREATE INDEX idx_ingest_jobs_created_at ON ingest_jobs(created_at DESC);

-- Audit logs
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(255) NOT NULL,
    resource_id UUID NOT NULL,
    details JSONB,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX idx_audit_logs_user ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);
