//! [FOAF](http://xmlns.com/foaf/spec/) (Friend of a Friend) vocabulary for describing people, their activities, and their relations to other people and objects.
//!
//! Source: <http://xmlns.com/foaf/spec/>

use oxixml_model::NamedNodeRef;

/// The `foaf` namespace IRI (`http://xmlns.com/foaf/0.1/`).
pub const NAMESPACE: &str = "http://xmlns.com/foaf/0.1/";

macro_rules! term {
    ($name:ident, $local:literal, $doc:expr) => {
        #[doc = $doc]
        pub const $name: NamedNodeRef<'static> =
            NamedNodeRef::new_unchecked(concat!("http://xmlns.com/foaf/0.1/", $local));
    };
}

term!(
    LABEL_PROPERTY,
    "LabelProperty",
    "A foaf:LabelProperty is any RDF property with texual values that serve as labels."
);
term!(PERSON, "Person", "A person.");
term!(
    AGENT,
    "Agent",
    "An agent (eg. person, group, software or physical artifact)."
);
term!(DOCUMENT, "Document", "A document.");
term!(ORGANIZATION, "Organization", "An organization.");
term!(GROUP, "Group", "A class of Agents.");
term!(
    PROJECT,
    "Project",
    "A project (a collective endeavour of some kind)."
);
term!(IMAGE, "Image", "An image.");
term!(
    PERSONAL_PROFILE_DOCUMENT,
    "PersonalProfileDocument",
    "A personal profile RDF document."
);
term!(ONLINE_ACCOUNT, "OnlineAccount", "An online account.");
term!(
    ONLINE_GAMING_ACCOUNT,
    "OnlineGamingAccount",
    "An online gaming account."
);
term!(
    ONLINE_ECOMMERCE_ACCOUNT,
    "OnlineEcommerceAccount",
    "An online e-commerce account."
);
term!(
    ONLINE_CHAT_ACCOUNT,
    "OnlineChatAccount",
    "An online chat account."
);
term!(
    MBOX,
    "mbox",
    "A personal mailbox, ie. an Internet mailbox associated with exactly one owner, the first owner of this mailbox. This is a 'static inverse functional property', in that there is (across time and change) at most one individual that ever has any particular value for foaf:mbox."
);
term!(
    MBOX_SHA1SUM,
    "mbox_sha1sum",
    "The sha1sum of the URI of an Internet mailbox associated with exactly one owner, the first owner of the mailbox."
);
term!(
    GENDER,
    "gender",
    "The gender of this Agent (typically but not necessarily 'male' or 'female')."
);
term!(
    GEEKCODE,
    "geekcode",
    "A textual geekcode for this person, see <http://www.geekcode.com/geek.html>"
);
term!(
    DNA_CHECKSUM,
    "dnaChecksum",
    "A checksum for the DNA of some thing. Joke."
);
term!(SHA1, "sha1", "A sha1sum hash, in hex.");
term!(
    BASED_NEAR,
    "based_near",
    "A location that something is based near, for some broadly human notion of near."
);
term!(TITLE, "title", "Title (Mr, Mrs, Ms, Dr. etc)");
term!(
    NICK,
    "nick",
    "A short informal nickname characterising an agent (includes login identifiers, IRC and other chat nicknames)."
);
term!(JABBER_ID, "jabberID", "A jabber ID for something.");
term!(AIM_CHAT_ID, "aimChatID", "An AIM chat ID");
term!(SKYPE_ID, "skypeID", "A Skype ID");
term!(ICQ_CHAT_ID, "icqChatID", "An ICQ chat ID");
term!(YAHOO_CHAT_ID, "yahooChatID", "A Yahoo chat ID");
term!(MSN_CHAT_ID, "msnChatID", "An MSN chat ID");
term!(NAME, "name", "A name for some thing.");
term!(FIRST_NAME, "firstName", "The first name of a person.");
term!(LAST_NAME, "lastName", "The last name of a person.");
term!(GIVEN_NAME, "givenName", "The given name of some person.");
term!(GIVENNAME, "givenname", "The given name of some person.");
term!(SURNAME, "surname", "The surname of some person.");
term!(
    FAMILY_NAME_UNDERSCORED,
    "family_name",
    "The family name of some person."
);
term!(FAMILY_NAME, "familyName", "The family name of some person.");
term!(
    PHONE,
    "phone",
    "A phone, specified using fully qualified tel: URI scheme (refs: <http://www.w3.org/Addressing/schemes.html#tel>)."
);
term!(HOMEPAGE, "homepage", "A homepage for some thing.");
term!(
    WEBLOG,
    "weblog",
    "A weblog of some thing (whether person, group, company etc.)."
);
term!(OPENID, "openid", "An OpenID for an Agent.");
term!(
    TIPJAR,
    "tipjar",
    "A tipjar document for this agent, describing means for payment and reward."
);
term!(
    PLAN,
    "plan",
    "A .plan comment, in the tradition of finger and '.plan' files."
);
term!(MADE, "made", "Something that was made by this agent.");
term!(MAKER, "maker", "An agent that made this thing.");
term!(
    IMG,
    "img",
    "An image that can be used to represent some thing (ie. those depictions which are particularly representative of something, eg. one's photo on a homepage)."
);
term!(DEPICTION, "depiction", "A depiction of some thing.");
term!(
    DEPICTS,
    "depicts",
    "A thing depicted in this representation."
);
term!(THUMBNAIL, "thumbnail", "A derived thumbnail image.");
term!(
    MYERS_BRIGGS,
    "myersBriggs",
    "A Myers Briggs (MBTI) personality classification."
);
term!(
    WORKPLACE_HOMEPAGE,
    "workplaceHomepage",
    "A workplace homepage of some person; the homepage of an organization they work for."
);
term!(
    WORK_INFO_HOMEPAGE,
    "workInfoHomepage",
    "A work info homepage of some person; a page about their work for some organization."
);
term!(
    SCHOOL_HOMEPAGE,
    "schoolHomepage",
    "A homepage of a school attended by the person."
);
term!(
    KNOWS,
    "knows",
    "A person known by this person (indicating some level of reciprocated interaction between the parties)."
);
term!(
    INTEREST,
    "interest",
    "A page about a topic of interest to this person."
);
term!(
    TOPIC_INTEREST,
    "topic_interest",
    "A thing of interest to this person."
);
term!(
    PUBLICATIONS,
    "publications",
    "A link to the publications of this person."
);
term!(
    CURRENT_PROJECT,
    "currentProject",
    "A current project this person works on."
);
term!(
    PAST_PROJECT,
    "pastProject",
    "A project this person has previously worked on."
);
term!(
    FUNDED_BY,
    "fundedBy",
    "An organization funding a project or person."
);
term!(LOGO, "logo", "A logo representing some thing.");
term!(TOPIC, "topic", "A topic of some page or document.");
term!(
    PRIMARY_TOPIC,
    "primaryTopic",
    "The primary topic of some page or document."
);
term!(
    FOCUS,
    "focus",
    "The underlying or 'focal' entity associated with some SKOS-described concept."
);
term!(
    IS_PRIMARY_TOPIC_OF,
    "isPrimaryTopicOf",
    "A document that this thing is the primary topic of."
);
term!(PAGE, "page", "A page or document about this thing.");
term!(THEME, "theme", "A theme.");
term!(
    ACCOUNT,
    "account",
    "Indicates an account held by this agent."
);
term!(
    HOLDS_ACCOUNT,
    "holdsAccount",
    "Indicates an account held by this agent."
);
term!(
    ACCOUNT_SERVICE_HOMEPAGE,
    "accountServiceHomepage",
    "Indicates a homepage of the service provide for this online account."
);
term!(
    ACCOUNT_NAME,
    "accountName",
    "Indicates the name (identifier) associated with this online account."
);
term!(MEMBER, "member", "Indicates a member of a Group");
term!(
    MEMBERSHIP_CLASS,
    "membershipClass",
    "Indicates the class of individuals that are a member of a Group"
);
term!(
    BIRTHDAY,
    "birthday",
    "The birthday of this Agent, represented in mm-dd string form, eg. '12-31'."
);
term!(AGE, "age", "The age in years of some agent.");
term!(
    STATUS,
    "status",
    "A string expressing what the user is happy for the general public (normally) to know about their current activity."
);

/// All 75 terms defined in the [`foaf`](self) module, in declaration order.
///
/// Exposed for introspection and exhaustiveness testing.
pub const ALL: &[NamedNodeRef<'static>] = &[
    LABEL_PROPERTY,
    PERSON,
    AGENT,
    DOCUMENT,
    ORGANIZATION,
    GROUP,
    PROJECT,
    IMAGE,
    PERSONAL_PROFILE_DOCUMENT,
    ONLINE_ACCOUNT,
    ONLINE_GAMING_ACCOUNT,
    ONLINE_ECOMMERCE_ACCOUNT,
    ONLINE_CHAT_ACCOUNT,
    MBOX,
    MBOX_SHA1SUM,
    GENDER,
    GEEKCODE,
    DNA_CHECKSUM,
    SHA1,
    BASED_NEAR,
    TITLE,
    NICK,
    JABBER_ID,
    AIM_CHAT_ID,
    SKYPE_ID,
    ICQ_CHAT_ID,
    YAHOO_CHAT_ID,
    MSN_CHAT_ID,
    NAME,
    FIRST_NAME,
    LAST_NAME,
    GIVEN_NAME,
    GIVENNAME,
    SURNAME,
    FAMILY_NAME_UNDERSCORED,
    FAMILY_NAME,
    PHONE,
    HOMEPAGE,
    WEBLOG,
    OPENID,
    TIPJAR,
    PLAN,
    MADE,
    MAKER,
    IMG,
    DEPICTION,
    DEPICTS,
    THUMBNAIL,
    MYERS_BRIGGS,
    WORKPLACE_HOMEPAGE,
    WORK_INFO_HOMEPAGE,
    SCHOOL_HOMEPAGE,
    KNOWS,
    INTEREST,
    TOPIC_INTEREST,
    PUBLICATIONS,
    CURRENT_PROJECT,
    PAST_PROJECT,
    FUNDED_BY,
    LOGO,
    TOPIC,
    PRIMARY_TOPIC,
    FOCUS,
    IS_PRIMARY_TOPIC_OF,
    PAGE,
    THEME,
    ACCOUNT,
    HOLDS_ACCOUNT,
    ACCOUNT_SERVICE_HOMEPAGE,
    ACCOUNT_NAME,
    MEMBER,
    MEMBERSHIP_CLASS,
    BIRTHDAY,
    AGE,
    STATUS,
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn namespace_ends_with_separator() {
        assert!(NAMESPACE.ends_with('#') || NAMESPACE.ends_with('/'));
    }

    #[test]
    fn terms_are_namespace_prefixed() {
        for term in ALL {
            assert!(
                term.as_str().starts_with(NAMESPACE),
                "{} is not prefixed by {}",
                term.as_str(),
                NAMESPACE
            );
            assert!(term.as_str().len() > NAMESPACE.len());
        }
    }

    #[test]
    fn no_duplicate_terms() {
        let mut seen = HashSet::new();
        for term in ALL {
            assert!(
                seen.insert(term.as_str()),
                "duplicate IRI: {}",
                term.as_str()
            );
        }
    }

    #[test]
    fn expected_term_count() {
        assert_eq!(ALL.len(), 75);
    }
}
