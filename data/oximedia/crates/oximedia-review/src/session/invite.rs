//! Session participant invitation.

use crate::{
    error::{ReviewError, ReviewResult},
    SessionId, User, UserRole,
};

/// Invitation status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvitationStatus {
    /// Invitation sent.
    Sent,
    /// Invitation accepted.
    Accepted,
    /// Invitation declined.
    Declined,
    /// Invitation expired.
    Expired,
}

/// Invitation details.
#[derive(Debug, Clone)]
pub struct Invitation {
    /// Session ID.
    pub session_id: SessionId,
    /// Invited user email.
    pub email: String,
    /// User role.
    pub role: UserRole,
    /// Invitation status.
    pub status: InvitationStatus,
}

/// Invite a participant to a review session.
///
/// # Arguments
///
/// * `session_id` - ID of the session
/// * `email` - Email address of the user to invite
/// * `role` - Role to assign to the user
///
/// # Errors
///
/// Returns error if invitation fails.
pub async fn invite_participant(
    session_id: SessionId,
    email: &str,
    role: UserRole,
) -> ReviewResult<()> {
    // Validate email format
    if !is_valid_email(email) {
        return Err(ReviewError::InvalidConfig(format!(
            "Invalid email address: {email}"
        )));
    }

    // In a real implementation, this would:
    // 1. Create an invitation record
    // 2. Send an email notification
    // 3. Store the invitation in a database

    let _invitation = Invitation {
        session_id,
        email: email.to_string(),
        role,
        status: InvitationStatus::Sent,
    };

    Ok(())
}

/// Invite multiple participants at once.
///
/// # Arguments
///
/// * `session_id` - ID of the session
/// * `invites` - List of (email, role) pairs
///
/// # Errors
///
/// Returns error if any invitation fails.
pub async fn invite_multiple(
    session_id: SessionId,
    invites: &[(String, UserRole)],
) -> ReviewResult<()> {
    for (email, role) in invites {
        invite_participant(session_id, email, *role).await?;
    }
    Ok(())
}

/// Accept an invitation.
///
/// # Arguments
///
/// * `session_id` - ID of the session
/// * `user` - User accepting the invitation
///
/// # Errors
///
/// Returns error if acceptance fails.
pub async fn accept_invitation(session_id: SessionId, user: User) -> ReviewResult<()> {
    // In a real implementation, this would:
    // 1. Update the invitation status
    // 2. Add the user to the session participants
    // 3. Send notification to session owner

    let _ = (session_id, user);
    Ok(())
}

/// Decline an invitation.
///
/// # Arguments
///
/// * `session_id` - ID of the session
/// * `email` - Email of the user declining
///
/// # Errors
///
/// Returns error if decline fails.
pub async fn decline_invitation(session_id: SessionId, email: &str) -> ReviewResult<()> {
    // In a real implementation, this would:
    // 1. Update the invitation status
    // 2. Send notification to session owner

    let _ = (session_id, email);
    Ok(())
}

fn is_valid_email(email: &str) -> bool {
    if let Some(at_pos) = email.find('@') {
        if let Some(dot_pos) = email.rfind('.') {
            return at_pos > 0 && dot_pos > at_pos + 1 && dot_pos < email.len() - 1;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_invite_participant() {
        let session_id = SessionId::new();
        let result =
            invite_participant(session_id, "reviewer@example.com", UserRole::Reviewer).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_invite_invalid_email() {
        let session_id = SessionId::new();
        let result = invite_participant(session_id, "invalid-email", UserRole::Reviewer).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_invite_multiple() {
        let session_id = SessionId::new();
        let invites = vec![
            ("user1@example.com".to_string(), UserRole::Reviewer),
            ("user2@example.com".to_string(), UserRole::Approver),
        ];
        let result = invite_multiple(session_id, &invites).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_is_valid_email() {
        assert!(is_valid_email("user@example.com"));
        assert!(!is_valid_email("invalid"));
        assert!(!is_valid_email("@example.com"));
    }

    #[test]
    fn test_invitation_status() {
        assert_eq!(InvitationStatus::Sent, InvitationStatus::Sent);
        assert_ne!(InvitationStatus::Sent, InvitationStatus::Accepted);
    }
}
