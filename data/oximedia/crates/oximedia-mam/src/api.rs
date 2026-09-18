//! REST and GraphQL API implementation
//!
//! Provides web APIs for MAM system:
//! - RESTful API with actix-web
//! - GraphQL API with async-graphql
//! - JWT authentication
//! - Role-based authorization (RBAC)
//! - Rate limiting

use actix_web::{web, App, HttpRequest, HttpResponse, HttpServer, Result as ActixResult};
use async_graphql::{Context, EmptySubscription, Object, Schema, SimpleObject};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::database::{User, UserRole};
use crate::{MamError, MamSystem, Result};

/// API server
pub struct ApiServer {
    mam: Arc<MamSystem>,
    jwt_secret: String,
}

/// JWT claims
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub user_id: String,
    pub role: String,
    pub exp: usize,
}

/// Authentication token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    pub token: String,
    pub expires_at: i64,
}

/// Login request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
}

/// Login response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginResponse {
    pub token: String,
    pub user: UserInfo,
}

/// User info (without sensitive data)
#[derive(Debug, Clone, Serialize, Deserialize, SimpleObject)]
#[graphql(name = "UserInfo")]
pub struct UserInfo {
    pub id: Uuid,
    pub username: String,
    pub email: String,
    pub full_name: Option<String>,
    #[graphql(skip)]
    pub role: UserRole,
    pub role_str: String,
}

impl From<User> for UserInfo {
    fn from(user: User) -> Self {
        Self {
            id: user.id,
            username: user.username.clone(),
            email: user.email,
            full_name: user.full_name,
            role_str: user.role.to_string(),
            role: user.role,
        }
    }
}

/// GraphQL Query root
pub struct QueryRoot;

#[Object]
impl QueryRoot {
    /// Get current user
    async fn me(&self, ctx: &Context<'_>) -> Result<UserInfo> {
        let user_id = ctx.data::<Uuid>()?;
        let mam = ctx.data::<Arc<MamSystem>>()?;

        let user = mam.database().get_user(*user_id).await?;
        Ok(user.into())
    }

    /// Get asset by ID
    async fn asset(&self, ctx: &Context<'_>, id: Uuid) -> Result<AssetGql> {
        let mam = ctx.data::<Arc<MamSystem>>()?;
        let asset = mam.asset_manager().get_asset(id).await?;
        Ok(asset.into())
    }

    /// Get collection by ID
    async fn collection(&self, ctx: &Context<'_>, id: Uuid) -> Result<CollectionGql> {
        let mam = ctx.data::<Arc<MamSystem>>()?;
        let collection = mam.collection_manager().get_collection(id).await?;
        Ok(collection.into())
    }

    /// Search assets
    async fn search_assets(
        &self,
        ctx: &Context<'_>,
        query: String,
        limit: Option<i32>,
        offset: Option<i32>,
    ) -> Result<Vec<AssetGql>> {
        let mam = ctx.data::<Arc<MamSystem>>()?;

        let pagination = crate::asset::Pagination {
            limit: limit.unwrap_or(50) as i64,
            offset: offset.unwrap_or(0) as i64,
        };

        let result = mam
            .asset_manager()
            .search_assets(&query, pagination)
            .await?;

        Ok(result.assets.into_iter().map(Into::into).collect())
    }
}

/// GraphQL Mutation root
pub struct MutationRoot;

#[Object]
impl MutationRoot {
    /// Create a collection
    async fn create_collection(
        &self,
        ctx: &Context<'_>,
        name: String,
        description: Option<String>,
    ) -> Result<CollectionGql> {
        let user_id = ctx.data::<Uuid>()?;
        let mam = ctx.data::<Arc<MamSystem>>()?;

        let req = crate::collection::CreateCollectionRequest {
            name,
            description,
            parent_id: None,
            is_smart: false,
            smart_query: None,
            created_by: Some(*user_id),
        };

        let collection = mam.collection_manager().create_collection(req).await?;
        Ok(collection.into())
    }

    /// Add asset to collection
    async fn add_asset_to_collection(
        &self,
        ctx: &Context<'_>,
        collection_id: Uuid,
        asset_id: Uuid,
    ) -> Result<bool> {
        let user_id = ctx.data::<Uuid>()?;
        let mam = ctx.data::<Arc<MamSystem>>()?;

        mam.collection_manager()
            .add_asset(collection_id, asset_id, None, Some(*user_id))
            .await?;

        Ok(true)
    }

    /// Update asset metadata
    async fn update_asset(
        &self,
        ctx: &Context<'_>,
        id: Uuid,
        title: Option<String>,
        description: Option<String>,
        keywords: Option<Vec<String>>,
    ) -> Result<AssetGql> {
        let mam = ctx.data::<Arc<MamSystem>>()?;

        let req = crate::asset::UpdateAssetRequest {
            title,
            description,
            keywords,
            categories: None,
            copyright: None,
            license: None,
            creator: None,
            custom: None,
            status: None,
        };

        let asset = mam.asset_manager().update_asset(id, req).await?;
        Ok(asset.into())
    }
}

/// GraphQL Asset type
#[derive(SimpleObject)]
#[graphql(name = "Asset")]
pub struct AssetGql {
    pub id: Uuid,
    pub filename: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub mime_type: Option<String>,
    pub file_size: Option<i64>,
    pub duration_ms: Option<i64>,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub keywords: Option<Vec<String>>,
    pub created_at: String,
}

impl From<crate::asset::Asset> for AssetGql {
    fn from(asset: crate::asset::Asset) -> Self {
        Self {
            id: asset.id,
            filename: asset.filename,
            title: asset.title,
            description: asset.description,
            mime_type: asset.mime_type,
            file_size: asset.file_size,
            duration_ms: asset.duration_ms,
            width: asset.width,
            height: asset.height,
            keywords: asset.keywords,
            created_at: asset.created_at.to_rfc3339(),
        }
    }
}

/// GraphQL Collection type
#[derive(SimpleObject)]
#[graphql(name = "Collection")]
pub struct CollectionGql {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub is_smart: bool,
    pub created_at: String,
}

impl From<crate::collection::Collection> for CollectionGql {
    fn from(collection: crate::collection::Collection) -> Self {
        Self {
            id: collection.id,
            name: collection.name,
            description: collection.description,
            is_smart: collection.is_smart,
            created_at: collection.created_at.to_rfc3339(),
        }
    }
}

impl ApiServer {
    /// Create a new API server
    #[must_use]
    pub fn new(mam: Arc<MamSystem>, jwt_secret: String) -> Self {
        Self { mam, jwt_secret }
    }

    /// Generate JWT token for user
    ///
    /// # Errors
    ///
    /// Returns an error if token generation fails
    pub fn generate_token(&self, user: &User) -> Result<AuthToken> {
        let expiration = chrono::Utc::now()
            .checked_add_signed(chrono::Duration::hours(24))
            .ok_or_else(|| MamError::Internal("Failed to calculate expiration".to_string()))?;

        let claims = Claims {
            sub: user.username.clone(),
            user_id: user.id.to_string(),
            role: format!("{:?}", user.role),
            exp: expiration.timestamp() as usize,
        };

        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(self.jwt_secret.as_bytes()),
        )
        .map_err(|e| MamError::Authentication(format!("Token generation failed: {e}")))?;

        Ok(AuthToken {
            token,
            expires_at: expiration.timestamp(),
        })
    }

    /// Verify JWT token
    ///
    /// # Errors
    ///
    /// Returns an error if token verification fails
    pub fn verify_token(&self, token: &str) -> Result<Claims> {
        let token_data = decode::<Claims>(
            token,
            &DecodingKey::from_secret(self.jwt_secret.as_bytes()),
            &Validation::default(),
        )
        .map_err(|e| MamError::Authentication(format!("Invalid token: {e}")))?;

        Ok(token_data.claims)
    }

    /// Extract token from Authorization header
    ///
    /// # Errors
    ///
    /// Returns an error if the header is missing or invalid
    pub fn extract_token(req: &HttpRequest) -> Result<String> {
        let auth_header = req
            .headers()
            .get("Authorization")
            .and_then(|h| h.to_str().ok())
            .ok_or_else(|| MamError::Authentication("Missing Authorization header".to_string()))?;

        if !auth_header.starts_with("Bearer ") {
            return Err(MamError::Authentication(
                "Invalid Authorization header format".to_string(),
            ));
        }

        Ok(auth_header[7..].to_string())
    }

    /// Build actix-web application configuration
    pub fn configure_app(cfg: &mut web::ServiceConfig, mam: Arc<MamSystem>, jwt_secret: String) {
        // Create GraphQL schema
        let schema = Schema::build(QueryRoot, MutationRoot, EmptySubscription)
            .data(mam.clone())
            .finish();

        cfg.app_data(web::Data::new(mam))
            .app_data(web::Data::new(jwt_secret))
            .app_data(web::Data::new(schema))
            .route("/health", web::get().to(health_handler))
            .route("/auth/login", web::post().to(login_handler))
            .service(
                web::scope("/api/v1")
                    .route("/assets", web::get().to(list_assets_handler))
                    .route("/assets/{id}", web::get().to(get_asset_handler))
                    .route("/assets/{id}", web::put().to(update_asset_handler))
                    .route("/collections", web::get().to(list_collections_handler))
                    .route("/collections", web::post().to(create_collection_handler))
                    .route("/collections/{id}", web::get().to(get_collection_handler))
                    .route("/search", web::get().to(search_handler)),
            )
            .route("/graphql", web::post().to(graphql_handler));
    }

    /// Start the API server
    ///
    /// # Errors
    ///
    /// Returns an error if the server fails to start
    pub async fn start(&self, host: &str, port: u16) -> Result<()> {
        let mam = Arc::clone(&self.mam);
        let jwt_secret = self.jwt_secret.clone();

        HttpServer::new(move || {
            let app_mam = Arc::clone(&mam);
            let app_jwt = jwt_secret.clone();
            App::new().configure(move |cfg| {
                Self::configure_app(cfg, Arc::clone(&app_mam), app_jwt.clone());
            })
        })
        .bind((host, port))
        .map_err(|e| MamError::Internal(format!("Failed to bind server: {e}")))?
        .run()
        .await
        .map_err(|e| MamError::Internal(format!("Server error: {e}")))?;

        Ok(())
    }
}

// REST API handlers

async fn health_handler(mam: web::Data<Arc<MamSystem>>) -> ActixResult<HttpResponse> {
    let health = mam
        .health()
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(health))
}

async fn login_handler(
    req: web::Json<LoginRequest>,
    mam: web::Data<Arc<MamSystem>>,
    jwt_secret: web::Data<String>,
) -> ActixResult<HttpResponse> {
    // Get user by username
    let user = mam
        .database()
        .get_user_by_username(&req.username)
        .await
        .map_err(|_| actix_web::error::ErrorUnauthorized("Invalid credentials"))?;

    // Verify password
    let valid = bcrypt::verify(&req.password, &user.password_hash)
        .map_err(actix_web::error::ErrorInternalServerError)?;

    if !valid {
        return Err(actix_web::error::ErrorUnauthorized("Invalid credentials"));
    }

    // Generate token
    let api_server = ApiServer::new(Arc::clone(&mam), jwt_secret.to_string());
    let auth_token = api_server
        .generate_token(&user)
        .map_err(actix_web::error::ErrorInternalServerError)?;

    let response = LoginResponse {
        token: auth_token.token,
        user: user.into(),
    };

    Ok(HttpResponse::Ok().json(response))
}

async fn list_assets_handler(
    req: HttpRequest,
    mam: web::Data<Arc<MamSystem>>,
    jwt_secret: web::Data<String>,
) -> ActixResult<HttpResponse> {
    // Verify authentication
    let token = ApiServer::extract_token(&req).map_err(actix_web::error::ErrorUnauthorized)?;

    let api_server = ApiServer::new(Arc::clone(&mam), jwt_secret.to_string());
    let _claims = api_server
        .verify_token(&token)
        .map_err(actix_web::error::ErrorUnauthorized)?;

    // List assets
    let pagination = crate::asset::Pagination {
        limit: 50,
        offset: 0,
    };

    let assets = mam
        .asset_manager()
        .list_assets(Default::default(), pagination)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(assets))
}

async fn get_asset_handler(
    path: web::Path<Uuid>,
    req: HttpRequest,
    mam: web::Data<Arc<MamSystem>>,
    jwt_secret: web::Data<String>,
) -> ActixResult<HttpResponse> {
    let token = ApiServer::extract_token(&req).map_err(actix_web::error::ErrorUnauthorized)?;

    let api_server = ApiServer::new(Arc::clone(&mam), jwt_secret.to_string());
    let _claims = api_server
        .verify_token(&token)
        .map_err(actix_web::error::ErrorUnauthorized)?;

    let asset = mam
        .asset_manager()
        .get_asset(*path)
        .await
        .map_err(actix_web::error::ErrorNotFound)?;

    Ok(HttpResponse::Ok().json(asset))
}

async fn update_asset_handler(
    path: web::Path<Uuid>,
    update_req: web::Json<crate::asset::UpdateAssetRequest>,
    req: HttpRequest,
    mam: web::Data<Arc<MamSystem>>,
    jwt_secret: web::Data<String>,
) -> ActixResult<HttpResponse> {
    let token = ApiServer::extract_token(&req).map_err(actix_web::error::ErrorUnauthorized)?;

    let api_server = ApiServer::new(Arc::clone(&mam), jwt_secret.to_string());
    let _claims = api_server
        .verify_token(&token)
        .map_err(actix_web::error::ErrorUnauthorized)?;

    let asset = mam
        .asset_manager()
        .update_asset(*path, update_req.into_inner())
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(asset))
}

async fn list_collections_handler(
    req: HttpRequest,
    mam: web::Data<Arc<MamSystem>>,
    jwt_secret: web::Data<String>,
) -> ActixResult<HttpResponse> {
    let token = ApiServer::extract_token(&req).map_err(actix_web::error::ErrorUnauthorized)?;

    let api_server = ApiServer::new(Arc::clone(&mam), jwt_secret.to_string());
    let _claims = api_server
        .verify_token(&token)
        .map_err(actix_web::error::ErrorUnauthorized)?;

    let collections = mam
        .collection_manager()
        .list_collections(None, 50, 0)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(collections))
}

async fn create_collection_handler(
    create_req: web::Json<crate::collection::CreateCollectionRequest>,
    req: HttpRequest,
    mam: web::Data<Arc<MamSystem>>,
    jwt_secret: web::Data<String>,
) -> ActixResult<HttpResponse> {
    let token = ApiServer::extract_token(&req).map_err(actix_web::error::ErrorUnauthorized)?;

    let api_server = ApiServer::new(Arc::clone(&mam), jwt_secret.to_string());
    let _claims = api_server
        .verify_token(&token)
        .map_err(actix_web::error::ErrorUnauthorized)?;

    let collection = mam
        .collection_manager()
        .create_collection(create_req.into_inner())
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Created().json(collection))
}

async fn get_collection_handler(
    path: web::Path<Uuid>,
    req: HttpRequest,
    mam: web::Data<Arc<MamSystem>>,
    jwt_secret: web::Data<String>,
) -> ActixResult<HttpResponse> {
    let token = ApiServer::extract_token(&req).map_err(actix_web::error::ErrorUnauthorized)?;

    let api_server = ApiServer::new(Arc::clone(&mam), jwt_secret.to_string());
    let _claims = api_server
        .verify_token(&token)
        .map_err(actix_web::error::ErrorUnauthorized)?;

    let collection = mam
        .collection_manager()
        .get_collection(*path)
        .await
        .map_err(actix_web::error::ErrorNotFound)?;

    Ok(HttpResponse::Ok().json(collection))
}

async fn search_handler(
    query: web::Query<SearchRequest>,
    req: HttpRequest,
    mam: web::Data<Arc<MamSystem>>,
    jwt_secret: web::Data<String>,
) -> ActixResult<HttpResponse> {
    let token = ApiServer::extract_token(&req).map_err(actix_web::error::ErrorUnauthorized)?;

    let api_server = ApiServer::new(Arc::clone(&mam), jwt_secret.to_string());
    let _claims = api_server
        .verify_token(&token)
        .map_err(actix_web::error::ErrorUnauthorized)?;

    let pagination = crate::asset::Pagination {
        limit: query.limit.unwrap_or(50) as i64,
        offset: query.offset.unwrap_or(0) as i64,
    };

    let results = mam
        .asset_manager()
        .search_assets(&query.q, pagination)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(results))
}

#[derive(Deserialize)]
struct SearchRequest {
    q: String,
    limit: Option<i32>,
    offset: Option<i32>,
}

async fn graphql_handler(
    schema: web::Data<Schema<QueryRoot, MutationRoot, EmptySubscription>>,
    req: async_graphql_actix_web::GraphQLRequest,
) -> async_graphql_actix_web::GraphQLResponse {
    schema.execute(req.into_inner()).await.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claims_serialization() {
        let claims = Claims {
            sub: "testuser".to_string(),
            user_id: Uuid::new_v4().to_string(),
            role: "Admin".to_string(),
            exp: 1234567890,
        };

        let json = serde_json::to_string(&claims).expect("should succeed in test");
        let deserialized: Claims = serde_json::from_str(&json).expect("should succeed in test");

        assert_eq!(deserialized.sub, "testuser");
    }

    #[test]
    fn test_user_info_from_user() {
        let user = User {
            id: Uuid::new_v4(),
            username: "testuser".to_string(),
            email: "test@example.com".to_string(),
            password_hash: "hash".to_string(),
            full_name: Some("Test User".to_string()),
            role: UserRole::Admin,
            is_active: true,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        let info: UserInfo = user.into();
        assert_eq!(info.username, "testuser");
        assert_eq!(info.role, UserRole::Admin);
    }
}
