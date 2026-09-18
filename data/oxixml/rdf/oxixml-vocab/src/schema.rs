//! [schema.org](https://schema.org/) — a curated subset of roughly 50 of the most commonly used core types and properties from the collaborative schema.org vocabulary. schema.org itself contains many hundreds of terms; this module is not exhaustive.
//!
//! Source: <https://schema.org/docs/full.html>

use oxixml_model::NamedNodeRef;

/// The `schema` namespace IRI (`http://schema.org/`).
pub const NAMESPACE: &str = "http://schema.org/";

macro_rules! term {
    ($name:ident, $local:literal, $doc:expr) => {
        #[doc = $doc]
        pub const $name: NamedNodeRef<'static> =
            NamedNodeRef::new_unchecked(concat!("http://schema.org/", $local));
    };
}

term!(
    THING,
    "Thing",
    "The most generic type of item in the schema.org type hierarchy."
);
term!(
    PERSON,
    "Person",
    "A person (alive, dead, undead, or fictional)."
);
term!(
    ORGANIZATION,
    "Organization",
    "An organization such as a school, NGO, corporation, club, etc."
);
term!(
    PLACE,
    "Place",
    "Entities that have a somewhat fixed, physical extension."
);
term!(
    POSTAL_ADDRESS,
    "PostalAddress",
    "The mailing address of a place or person."
);
term!(
    GEO_COORDINATES,
    "GeoCoordinates",
    "The geographic coordinates of a place or event."
);
term!(
    CREATIVE_WORK,
    "CreativeWork",
    "The most generic kind of creative work, including books, movies, photographs, software programs, etc."
);
term!(
    ARTICLE,
    "Article",
    "An article, such as a news article or piece of investigative report."
);
term!(BLOG_POSTING, "BlogPosting", "A blog post.");
term!(
    NEWS_ARTICLE,
    "NewsArticle",
    "An article whose content reports news, or provides background context and supporting materials for understanding the news."
);
term!(BOOK, "Book", "A book.");
term!(
    WEB_PAGE,
    "WebPage",
    "A web page, typically described in more detail as the specific type of page, e.g. an about page, contact page, and so on."
);
term!(WEB_SITE, "WebSite", "A website.");
term!(IMAGE_OBJECT, "ImageObject", "An image file.");
term!(VIDEO_OBJECT, "VideoObject", "A video file.");
term!(
    EVENT,
    "Event",
    "An event happening at a certain time and location, such as a concert, lecture, or festival."
);
term!(PRODUCT, "Product", "Any offered product or service.");
term!(
    OFFER,
    "Offer",
    "An offer to transfer some rights to an item or to provide a service, for example an offer to sell a product, rent a movie, or perform a service."
);
term!(
    REVIEW_CLASS,
    "Review",
    "A review of an item, such as a product, service, or business."
);
term!(
    RATING,
    "Rating",
    "A rating is an evaluation on a numeric scale, such as 1 to 5 stars."
);
term!(
    AGGREGATE_RATING_CLASS,
    "AggregateRating",
    "The average rating based on multiple ratings or reviews."
);
term!(RECIPE, "Recipe", "A recipe, e.g. for a meal or a cocktail.");
term!(
    LOCAL_BUSINESS,
    "LocalBusiness",
    "A particular physical business or branch of an organization, usually located at a particular, fixed geographic location."
);
term!(
    ACTION,
    "Action",
    "An action performed by a direct agent and indirect participants upon a direct object."
);
term!(NAME, "name", "The name of the item.");
term!(DESCRIPTION, "description", "A description of the item.");
term!(URL, "url", "URL of the item.");
term!(IMAGE, "image", "An image of the item.");
term!(
    IDENTIFIER,
    "identifier",
    "The identifier property represents any kind of identifier for any kind of Thing, such as ISBNs, GTIN codes, UUIDs, etc."
);
term!(
    SAME_AS,
    "sameAs",
    "URL of a reference web page that unambiguously indicates the item's identity, e.g. a Wikipedia or Wikidata page."
);
term!(AUTHOR, "author", "The author of this content.");
term!(
    CREATOR,
    "creator",
    "The creator/author of this CreativeWork."
);
term!(
    PUBLISHER,
    "publisher",
    "The publisher of the creative work."
);
term!(
    DATE_CREATED,
    "dateCreated",
    "The date on which the item was created."
);
term!(
    DATE_PUBLISHED,
    "datePublished",
    "Date of first broadcast/publication."
);
term!(
    DATE_MODIFIED,
    "dateModified",
    "The date on which the item was most recently modified."
);
term!(
    KEYWORDS,
    "keywords",
    "Keywords or tags used to describe some item."
);
term!(
    IN_LANGUAGE,
    "inLanguage",
    "The language of the content or performance, or a work performed or provided by some event, in the form of a BCP 47 language tag."
);
term!(ADDRESS, "address", "Physical address of the item.");
term!(TELEPHONE, "telephone", "The telephone number.");
term!(EMAIL, "email", "Email address.");
term!(LOGO, "logo", "An associated logo.");
term!(
    CONTACT_POINT,
    "contactPoint",
    "A contact point for a person or organization."
);
term!(
    AGGREGATE_RATING,
    "aggregateRating",
    "The overall rating, based on a collection of reviews or ratings, of the item."
);
term!(REVIEW, "review", "A review of the item.");
term!(
    OFFERS,
    "offers",
    "An offer to sell, or provide services related to, this item."
);
term!(
    PRICE,
    "price",
    "The offer price of a product, or of a price component when attached to PriceSpecification and its subtypes."
);
term!(
    PRICE_CURRENCY,
    "priceCurrency",
    "The currency of the price, in three-letter ISO 4217 format."
);
term!(
    AVAILABILITY,
    "availability",
    "The availability of this item, e.g. `InStock`, `OutOfStock`, `PreOrder`, etc."
);
term!(
    BRAND,
    "brand",
    "The brand(s) associated with a product or service."
);

/// All 50 terms defined in the [`schema`](self) module, in declaration order.
///
/// Exposed for introspection and exhaustiveness testing.
pub const ALL: &[NamedNodeRef<'static>] = &[
    THING,
    PERSON,
    ORGANIZATION,
    PLACE,
    POSTAL_ADDRESS,
    GEO_COORDINATES,
    CREATIVE_WORK,
    ARTICLE,
    BLOG_POSTING,
    NEWS_ARTICLE,
    BOOK,
    WEB_PAGE,
    WEB_SITE,
    IMAGE_OBJECT,
    VIDEO_OBJECT,
    EVENT,
    PRODUCT,
    OFFER,
    REVIEW_CLASS,
    RATING,
    AGGREGATE_RATING_CLASS,
    RECIPE,
    LOCAL_BUSINESS,
    ACTION,
    NAME,
    DESCRIPTION,
    URL,
    IMAGE,
    IDENTIFIER,
    SAME_AS,
    AUTHOR,
    CREATOR,
    PUBLISHER,
    DATE_CREATED,
    DATE_PUBLISHED,
    DATE_MODIFIED,
    KEYWORDS,
    IN_LANGUAGE,
    ADDRESS,
    TELEPHONE,
    EMAIL,
    LOGO,
    CONTACT_POINT,
    AGGREGATE_RATING,
    REVIEW,
    OFFERS,
    PRICE,
    PRICE_CURRENCY,
    AVAILABILITY,
    BRAND,
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
        assert_eq!(ALL.len(), 50);
    }
}
