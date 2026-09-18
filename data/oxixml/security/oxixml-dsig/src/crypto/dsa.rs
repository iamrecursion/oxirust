//! DSA — [FIPS 186-4 §4.6–4.7](https://doi.org/10.6028/NIST.FIPS.186-4).
//!
//! The signature is a pair `(r, s)` of integers below the subgroup order `q`,
//! and [XML Signature §6.4.1](https://www.w3.org/TR/xmldsig-core1/#sec-DSA)
//! transmits it the same way ECDSA does: the plain concatenation of two `I2OSP`
//! encodings of a fixed width, never ASN.1. The width is fixed *by the algorithm
//! identifier*, not by the key — 20 octets for `#dsa-sha1` and 32 for
//! `xmldsig11#dsa-sha256` — so a key whose `q` is narrower than that (the
//! `L = 2048, N = 224` pairing FIPS 186-4 allows) still produces 32-octet halves
//! with leading zeros. The caller passes the width, having checked with
//! [`DsaPublicKey::fits_signature_width`](crate::DsaPublicKey::fits_signature_width)
//! that the key's `q` fits in it.
//!
//! Verification is the textbook computation:
//!
//! ```text
//! w  = s⁻¹ mod q
//! u₁ = z·w mod q          z = leftmost min(N, hashlen) bits of H(M)
//! u₂ = r·w mod q
//! v  = ((g^u₁ · y^u₂) mod p) mod q      accept iff v == r
//! ```
//!
//! Signing uses the deterministic nonce of
//! [RFC 6979](https://www.rfc-editor.org/rfc/rfc6979), for the reasons that
//! module explains: a repeated `k` publishes the private key, and DSA has no
//! recovery from that.
//!
//! # Timing
//!
//! The modular exponentiations are [`BigUint::modpow`], which is not constant
//! time. Verification is over public data; signing exposes the nonce and the
//! private key to an attacker who can measure the arithmetic, which is the same
//! posture as the ECDSA path and is documented on the public API.

use oxixml_support::bigint::BigUint;

use crate::algorithm::DigestAlgorithm;
use crate::crypto::digest;
use crate::crypto::rfc6979::{NonceGenerator, bits2int};

/// How many nonces are tried before signing gives up; see
/// [`crate::crypto::ecdsa`] for why this can only be reached by a degenerate key.
const MAX_SIGNATURE_ATTEMPTS: usize = 32;

/// The domain parameters and public value of a DSA key, as the arithmetic sees
/// them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PublicParameters<'a> {
    /// The prime modulus `p`.
    pub(crate) p: &'a BigUint,
    /// The subgroup order `q`.
    pub(crate) q: &'a BigUint,
    /// The generator `g` of the order-`q` subgroup.
    pub(crate) g: &'a BigUint,
    /// The public value `y = g^x mod p`.
    pub(crate) y: &'a BigUint,
}

/// Reduces a digest to the scalar `z` of FIPS 186-4 §4.7.
fn message_scalar(order: &BigUint, algorithm: DigestAlgorithm, message: &[u8]) -> Option<BigUint> {
    let hash = digest::digest(algorithm, message);
    Some(bits2int(&hash, order.bit_len()).div_rem(order)?.1)
}

/// DSA verification.
///
/// `half_width` is the octet width XML Signature fixes for each half of the
/// signature value; anything of a different total length is rejected outright.
#[must_use]
pub(crate) fn verify(
    key: PublicParameters<'_>,
    algorithm: DigestAlgorithm,
    message: &[u8],
    signature: &[u8],
    half_width: usize,
) -> bool {
    let (Some(r_octets), Some(s_octets)) = (
        signature.get(..half_width),
        signature.get(half_width..2 * half_width),
    ) else {
        return false;
    };
    if signature.len() != 2 * half_width {
        return false;
    }
    let r = BigUint::from_be_bytes(r_octets);
    let s = BigUint::from_be_bytes(s_octets);
    // FIPS 186-4 §4.7 step 1: both components must lie in 0 < · < q.
    if r.is_zero() || s.is_zero() || &r >= key.q || &s >= key.q {
        return false;
    }

    let (Some(z), Some(w)) = (
        message_scalar(key.q, algorithm, message),
        s.mod_inverse(key.q),
    ) else {
        return false;
    };
    let (Some(u1), Some(u2)) = (z.mul_mod(&w, key.q), r.mul_mod(&w, key.q)) else {
        return false;
    };
    let (Some(left), Some(right)) = (key.g.modpow(&u1, key.p), key.y.modpow(&u2, key.p)) else {
        return false;
    };
    let Some(product) = left.mul_mod(&right, key.p) else {
        return false;
    };
    let Some((_, v)) = product.div_rem(key.q) else {
        return false;
    };
    v == r
}

/// DSA signing with the deterministic nonce of RFC 6979.
///
/// Returns the raw `r ‖ s` of XML Signature §6.4.1, or [`None`] when the private
/// value is not in `1..q` or the arithmetic runs into an operand wider than
/// [`BigUint::MAX_BITS`].
#[must_use]
pub(crate) fn sign(
    p: &BigUint,
    q: &BigUint,
    g: &BigUint,
    private_key: &BigUint,
    algorithm: DigestAlgorithm,
    message: &[u8],
    half_width: usize,
) -> Option<Vec<u8>> {
    if private_key.is_zero() || private_key >= q {
        return None;
    }
    let hash = digest::digest(algorithm, message);
    let z = bits2int(&hash, q.bit_len()).div_rem(q)?.1;
    let mut nonces = NonceGenerator::new(algorithm, q, private_key, &hash)?;

    for _ in 0..MAX_SIGNATURE_ATTEMPTS {
        let k = nonces.next_nonce(q)?;
        let r = g.modpow(&k, p)?.div_rem(q)?.1;
        if r.is_zero() {
            continue;
        }
        let s = {
            let k_inverse = k.mod_inverse(q)?;
            let masked = r.mul_mod(private_key, q)?;
            k_inverse.mul_mod(&z.add_mod(&masked, q)?, q)?
        };
        if s.is_zero() {
            continue;
        }
        let mut value = r.to_be_bytes_padded(half_width)?;
        value.extend_from_slice(&s.to_be_bytes_padded(half_width)?);
        return Some(value);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkey::hex;

    /// The worked example of FIPS 186-2 Appendix 5, which XML Signature §6.4.1
    /// reproduces as *its* example of a `SignatureValue`: a 512-bit key with a
    /// 160-bit `q`, signing the three octets `abc` with SHA-1.
    const EXAMPLE_P: &str = "\
        8DF2A494492276AA3D25759BB06869CBEAC0D83AFB8D0CF7CBB8324F0D7882E5\
        D0762FC5B7210EAFC2E9ADAC32AB7AAC49693DFBF83724C2EC0736EE31C80291";
    const EXAMPLE_Q: &str = "C773218C737EC8EE993B4F2DED30F48EDACE915F";
    const EXAMPLE_G: &str = "\
        626D027839EA0A13413163A55B4CB500299D5522956CEFCB3BFF10F399CE2C2E\
        71CB9DE5FA24BABF58E5B79521925C9CC42E9F6F464B088CC572AF53E6D78802";
    const EXAMPLE_X: &str = "2070B3223DBA372FDE1C0FFC7B2E3B498B260614";
    const EXAMPLE_Y: &str = "\
        19131871D75B1612A819F29D78D1B0D7346F7AA77BB62A859BFD6C5675DA9D21\
        2D3A36EF1672EF660B8C7C255CC0EC74858FBA33F44C06699630A76B030EE333";
    const EXAMPLE_R: &str = "8BAC1AB66410435CB7181F95B16AB97C92B341C0";
    const EXAMPLE_S: &str = "41E2345F1F56DF2458F426D155B4BA2DB6DCD8C8";

    /// Domain parameters of the CAVP FIPS 186-3 `SigVer` group
    /// `[mod = L=1024, N=160, SHA-1]`.
    const CAVP_1024_SHA1_P: &str = "\
        DC5BF3A88B2D99E4C95CDD7A0501CC38630D425CF5C390AF3429CFF1F35147B7\
        95CAEA923F0D3577158F8A0C89DABD1962C2C453306B5D70CACFB01430ACEB54\
        E5A5FA6F9340D3BD2DA612FCEEB76B0EC1EBFAE635A56AB141B108E00DC76EEF\
        E2EDD0C514C21C457457C39065DBA9D0ECB7569C247172D8438AD2827B60435B";
    const CAVP_1024_SHA1_Q: &str = "E956602B83D195DBE945B3AC702FC61F81571F1D";
    const CAVP_1024_SHA1_G: &str = "\
        D7EB9CA20A3C7A079606BAFC4C9261CCABA303A5DC9FE9953F197DFE548C2348\
        95BAA77F441EE6A2D97B909CBBD26FF7B869D24CAE51B5C6EDB127A4B5D75CD8\
        B46608BFA148249DFFDB59807C5D7DDE3FE3080CA3A2D28312142BECB1FA8E24\
        003E21C7287108174B95D5BC711E1C8D9B1076784F5DC37A964A5E51390DA713";

    /// `(message, y, r, s, expected)` from that group.
    const CAVP_1024_SHA1_CASES: [(&str, &str, &str, &str, bool); 6] = [
        (
            "\
            0FE1BFEE500BDB76026099B1D37553F6BDFE48C82094EF98CB309DD777330BED\
            FAA2F94C823EF74EF4074B50D8706041AC0E371C7C22DCF70263B8D60E17A86C\
            7C379CFDA8F22469E0DF9D49D59439FC99891873628FFF25DDA5FAC5AC794E94\
            8BABDDE968143BA05F1128F34FDAD5875EDC4CD71C6C24BA2060FFBD439CE2B3",
            "\
            6240EA0647117C38FE705106D56DB578F3E10130928452D4F3587881B8A2BC68\
            73A8BEFC3237F20914E2A91C7F07A928EE22ADEED23D74AB7F82EA11F70497E5\
            78F7A9B4CBD6F10226222B0B4DA2EA1E49813D6BB9882FBF675C0846BB80CC89\
            1857B89B0EF1BEB6CCE3378A9AAB5D66AD4CB9277CF447DFE1E64434749432FB",
            "B5AF307867FB8B54390013CC67020DDF1F2C0B81",
            "620D3B22AB5031440C3E35EAB6F481298F9E9F08",
            true,
        ),
        (
            "\
            97D50898025D2F9BA633866E968CA75E969D394EDBA6517204CB3DD537C2BA38\
            778A2DC9DBC685A915E5676FCD43BC3726BC59CE3D7A9FAE35565082A069C139\
            FA37C90D922B126933DB3FA6C5EF6B1EDF00D174A51887BB76909C6A94FE994E\
            CC7B7FC8F26113B17F30F9D01693DF99A125B4F17E184331C6B6E8CA00F54F3A",
            "\
            69974DE550FE6BD3099150FAEA1623AD3FB6D9BF23A07215093F319725AD0877\
            ACCFFD291B6DA18EB0CBE51676CEB0977504EB97C27C0B191883F72FB2710A9F\
            BD8BCF13BE0BF854410B32F42B33EC89D3CC1CF892BCD536C4195CA9ADA302AD\
            600C3408739935D77DC247529CA47F844CC86F5016A2FE962C6E20CA7C4D4E8F",
            "B5D05FAA7005764E8DAE0327C5BF1972FF7681B9",
            "18EA15BD9F00475B25204CBC23F8C23E01588015",
            false,
        ),
        (
            "\
            09560B84EC9A6ADD0C1C08CDE95CB15EF104D34EAE2CD029B126E5EBA56F96B0\
            F369E0155625A984E3478491BBA851D132995F52E5B7343161D5D8DCB708289F\
            FC576F094D2AA91C5C7F2B64C9D1012F370965E5F38268AAFA8CC1DA79CCE7C2\
            3B03BF6774339D0DFCB451D606A180496FD4C3E528F311111203AB3BB3B8D3F4",
            "\
            34EC505F1CF37DE3FDF23806C5D10E26684B56B302E72033F01E4D8443ECC027\
            57E048848FF94A6677F6756FACB2243E264F8A4ED90DC0B0F2659CCFBBAFB4E9\
            95EE8F8ED85C908EE5CB05BE6B9DE971867F8EC9B379775C0DE88787BB6069EE\
            746331497030F7FE47052EB3DEC058A2AC4109C6B3F5EA23786D12BBAEFF1B14",
            "897F9B11E30C7D14652A3E8095554C70A8EAF02F",
            "79C2CC23A2AD7FB1A79F3A2D78084C52DA57628E",
            false,
        ),
        (
            "\
            7BC0C8B0FE1AE18EDF1F85BF770A4A1AB23CA56C0D1A05C5C919CBFB256B4C53\
            5A8C8725A880DE39E6713213168C59C7A9222778D96C8A0113FBC573DB90885D\
            106DD3D9AE9FB4C9E832CBD49CDC6FFFAD50D453AD1E90D96CD83363F94BA93C\
            CDAE50940F043E93E8640A9B225A0E46A0FAA965311D7E11D09CCD3FBFEE6564",
            "\
            4DD48B01D12F634EDED08BAD2A7DF0C908324FA690BFE976B28D7B6817162CA7\
            37B52DE12DD0D6C4E2570C8F35EDE2BCE0C6B42E751BF9E5A29CC16D04013B0C\
            A992D52198C7571D96B5783E7302359FA08405B5EAAFC4309E4FA8C626B704E7\
            6A8C83187295EA3A34E195C5E01080F43EC11191B064EFF5D86228BCA471966B",
            "BB35F94025B0291DBF9359446DFCF219729E04AD",
            "909B811CF541506A9FFE4C7EF936D1899B8B4D27",
            false,
        ),
        (
            "\
            E054512BBF008ECE2E404B311ED3EC1E1EB4EDD5F751F1C6AE261D1823F5B9D9\
            BC742ED692FC0B05DC92E29A07BFAB3CFC855D19D2C0035757A04ECA5E225381\
            4DAB908276D85B3A25A8244C190C72B5E8025DC99A4EE34BA81FB5CDF01578E5\
            D8BDAF41070D24AA8A82D0AB115056475A16988D56738224138E0A055850AB82",
            "\
            AAFAFF8C7DA2018EAE7766354A2FE1AAFE9C7895613B2671517480E495EE057A\
            3BDB1BA1628075676BCB2196A8780DEBD373A992DC11AD5C9332942F03C2D549\
            2C34175FC42DF2EBCA4C8FB2EB2C06A2A15AECED0AE7B05383A25D6636B726C3\
            20BE7EA82C698374AECD0BB76826848A5145F202C9DF13D880877572438C622B",
            "6D866ACF82F76000C82811C533E94F5CF31B49B9",
            "B3A338F22649FDF98B72E4FCE357A916F3FF60B7",
            false,
        ),
        (
            "\
            B966012B318DAFBE4BA7C94E6EBF59165A1EBC5D5A4B279A2E314A502860513E\
            AF1D1DBC1C10A8AA519B204006244E53E83451BDBE9DEA8C922387BD5B2E59B6\
            1270E31B26EC2C34996C2BDDCB85AD01F1C23D4F1ED716D2A179189BE0DEE32F\
            2F71CE6FAE80019F8009C55B6E4FC9B215A4687A776087A6C33E5CE5D564EF11",
            "\
            3D9DCB5A51F4A5C3AC315A255FC2D1F8F9B2DA4921A057403D31C4A4E000BFEC\
            C4ED5AA7432C1791073B46ED8781993486D0B8CF8C5674D3EC87E03FBCC228C7\
            510996FEDEF4B42E69788ADC771920EA75991E24C14CCE9B3E703512729A75D6\
            AC2CEBD513CC145F03548D5FD8DDEF23932C7B39AD4F2DB50F6A304121453B4C",
            "60B989352FBE35BE141B0DB29C9DD22B55B65FD1",
            "20B375C40E01300209CCF2E3DB0681ECCB03B64A",
            false,
        ),
    ];

    /// Domain parameters of the CAVP FIPS 186-3 `SigVer` group
    /// `[mod = L=2048, N=256, SHA-256]`.
    const CAVP_2048_SHA256_P: &str = "\
        CDF428329E226CF715F18EED005E439A8C7B927EDD24C866BE6C1B370057059E\
        A426D06F584F8E3C89F02FE8D4042604A2FE0DB63A87FF018DFAEC7790B88FD1\
        DA8396561AE62DF6F18D3540992EFC5ECCE63068F5F595687B8BBEED5801D5B6\
        C6BDF362DADEBB80E190D719D144DB693FC43CEFBD72B149570A96282FD9441C\
        397D98B15D73F8CFAA8F6514A16F5992A0A02FD7B6E4932B1E7E7EF2DB717815\
        B11E867E187AAE26F9C16CA0A5CA434ACF8F3356C3711765FE5E548B1EDCE381\
        BDD843580F9A881D702F00A0719C3BFA8576304DECB616B08EF8DB6F8C3D4892\
        3A2A30B9B505AD737AF1CBB0019558487400379738FD89C1EE01285E76E2D98B";
    const CAVP_2048_SHA256_Q: &str =
        "DAF3EBD85F57A8731538596876E8A34E73C732BC69B4D64B010BD7D5B63A09EF";
    const CAVP_2048_SHA256_G: &str = "\
        6CB6A8C1B8EA97173E5AE1D2A2D530468FD932B81A4F3C3E11042D56A61A504B\
        9C207DA7CDCA293C04F78583CF218E1CD2C1A92637AB4FF61D2EAA5E8E8221BF\
        E17E6A741791F21C799221AE4C703262A6DC2E295E36939E248DACAAFFE67307\
        1F6DC7A90D7CA3427556FB99FB1FDCA9DE3F745B5E56BE5B5933EBCBA4F0C60F\
        EF3798CDCA519997EF74ACD7000D3A286D01C7000A66FE0066BE4381842AA040\
        C3BF465306C38752D41C29162C4D294F2B1F7E9AD9232AADB469047F4050B085\
        BAFA06283DEF7D84958528A3F129A3BC81466C0CCD493FCAA87E805E7BD44500\
        90CE55652DBA15271234FBE0CFEF66F3ED9D569E565BD1218F2856637A5B26A4";

    /// `(message, y, r, s, expected)` from that group.
    const CAVP_2048_SHA256_CASES: [(&str, &str, &str, &str, bool); 6] = [
        (
            "\
            BC3414673A2A567AAD487AF1385B90BD1ADB92361F59E27EDE8AF63E964ECE90\
            B1AA7A330A8E41549E3D1E0CD32B339FBAF1120A79CA3768BD1F60BEA4DEF3EA\
            A4625CB9B19169248800F82837FEB151FD5D24DECE7BB1F7DE45175CF77DFADA\
            A12AE77F3F2CF55DDFD38714892D556EC91886C9734842304F1BE1AF439A56E8",
            "\
            002CC1264427522AA6FF6C9E3BEB23E1D1714E7A3CD3B592FAA74FF44E8D81A1\
            8B416C887A97A6E2F4592529116090A6EC4F0E73A45A9984C483CF7206E302C7\
            D77E27E5EB11AC581821BD2B91B9857764E155F850F5C25B909D6FE23C19595D\
            DE40750C7FB0E62D95C2944081C0D10767A11DBEB092C9B08D6D197DD34CDBD8\
            73F8DB7C3D33987ED2FD2E395C86588EDF131FADB936EF8085741F28C8F9D784\
            8C65A709F9C44B8DB85989732332D809544C8C5362B39F97331B66086AF5D3DC\
            703A2C30D96EF0763E13C171728D7B057B302C0C04642AA77B896C743DA9F0D3\
            0349745E2E552BAC2A4178591019AF33058A1948AE4059C2EFD7AA9F05DF06BB",
            "59F9E094D7D55D3A3A362C6671542A92034E89F36354DA25BD4F3EFD65B430EF",
            "47A78BA3F773F398361C41266CE6C84D0443B9B233B52C7939B9DC6D69051E9C",
            false,
        ),
        (
            "\
            F93230C8E56F77D8507CADB3D3B930C715E06763F2F38CC735EC626E650E4D3B\
            76B4F0427500106DAA2671DCEBBFAF04A8C660FA285CC059894C685B5EB0098D\
            12E406A47200C66818C255A7A55EDBB4D0A94FEF563AF9330BD90286E9A18591\
            0D741C19BE9965F44B630FBBB5F434967101CCBC693545A39430402CC0DC270C",
            "\
            1C3AA2F539438145B05DD13A17D666022944FE6E3DC2A14E41A32FBC85546E0D\
            A88B5B9D822A17F0C30ED317F4CC716F2847871C192EA4DDA488AEA291EC31FB\
            5A73424D12B034B642E175F25B102217A0BF1DE695923900C87F3F2911107EE0\
            6BB9F58CEF4D18C29A016EBF63A4BEA851C3A7E85A7C3D69E8896F59F9C7F1D7\
            7386C1E80391A1B9C4F82EAF59B6D72C7E8663B121FDD3B5510D5DB5971658D2\
            33E25B646A27B1E1AE0C859DC73F7763B91FFF6B3140E284C3A4650D17BEECA2\
            319C5C4A545F6DE728137F5BBAC6E3027E5588294AE5532891B8053DC9CBD044\
            9A09E54F4E3AC4EAF7E1BD2A727DE8F14A001DB92870807898BB504F5F4389BC",
            "96246546C49D2A00C5605CC5BCB18E229E272374C4D7FA6C8ACE99EAFC080342",
            "B77D5DD30B6DAAE84580CF2B3CF31ECA713C884DD5659CF5307C91E525991487",
            true,
        ),
        (
            "\
            7B35288D90892F2E42E8FAF3EBCFE41F268C03A76BE3A780EF8EAC9D00A41C13\
            79145C6B4A5CAED00DDE973EE9CD08E1945E55240F3AE85B78595A2F9365E7A2\
            A31359816BF230595FEF779E0092C978399A3E9DCD3ECAF284B4A64AC7FBBE77\
            342FC58842C7710B431096A1A0C0F44273B98077C41D5B44905C6E8DA47BEB0E",
            "\
            4EA940B3DC8E2622114E1820E2FB96B0148088F99AB46BBADC0FA70D6A8F20BD\
            27E7B12B13DFBB16D8A604B94CA03678E2D47CF0968C39CAB9C300361033FFC2\
            E2A4F37DFE1D9F16CD9B4031BC643A17C13C374CFCF398CE70934E7D7BDEDB92\
            1C6AD84D8BD40ED3B0A752A66BE1D1F6CA3B1609123D33CE1D02598B1E401F24\
            04B01820A6B66D30E7F27EB975B6B5683335398BA40CC412341FEB520D79048F\
            D5507141C7B105462C868CD2F366F7887F25C6F221E4272D7FB05D5DED5C7B1B\
            E290A11C5B0AC7CF1E045A5B6CDF8912C4CA4085F064F2FCC95D45FDBE7D90D8\
            D5E68AFF9CA0DF11E23FE2EA8BE23456BD499DD4BFF5309D3EC10D7607E5EE32",
            "469B70A4A37FA58EB9AA7E110273D8304F7C3DCD7BA399D519E69815972D397E",
            "25B848BFD4F7712CABBF3596FBFE922D0337486186E8F8EA9133E55F5F7F30EA",
            false,
        ),
        (
            "\
            3BF2CF1BED29D85904F79ADE4F945ABB91DBF7A83BAB4CBE0BD753CED2E40B77\
            F0297325FAC5172D55001985B79046AD29ACE15F08DAC7D27454EF67214258EF\
            F7F03F46DA19A4C3F12A90DF03F18098C21E92CE4E1AE413516DC9FFCFE2C323\
            7AB98DD344DD2E7A86011C8BA3F9DB67D35A3379485FCB3FC9B7C5DDE1CE2593",
            "\
            ABFE91C1E595BF644236DA5C4B2D1E8D894DA076A7060309C09FE3CC27B9BE06\
            30E53602401E41FF9E2DD86C3224DB018004127B7088EE92498E6214A31C5733\
            F9BAC4BDA83285A8CD87B8D55C3A92DD9BA7BC6F4C8CE1718578854495DBEF8B\
            F5F4D8EF7BFA5693300A02F25A8EBDDB64F3B32CE179A2663B1BA618EFEE5D05\
            F9DE33B7ED49AEA5286873963EA69974A1ADC11FA734CC8D1655CC95BE5C5A0C\
            70883C3AD715B879807B575627E8E4C101AAC7CBB1D1F11276CAF8AE9C6500B4\
            A2EB3B68FC3A76600EC34540D63134B07EF01324DDC33E1DEFB63EE20AA90B82\
            3696DF2998B9613A4A1485FB6966D6FADEEC6E78B21D6AB06C2B1BD8AA84921B",
            "54961CFD862CBC6D1F378AD00CC786F64A6763CF248DD55A95F413111F4626CF",
            "54DD5F1064D8221AE2315D3106724A2C34AB1B4E1F95DBC1D74792A648E5D573",
            false,
        ),
        (
            "\
            B24BC74E5696AC80E052B7A0CDA7B093FE8E4CB441A43CCB92F6967898986783\
            C6BB50816521C28C441F8DC82BF211DBBE709D69C75131529C15AD256BCEA261\
            7438D548777B4BA4B322DC42944BC759EEE5D497A6A9E02C33831A4169931C3C\
            3FDF4C285F71CEDE652A302659F4DD19951A3D0F48B6F16C831CE13877E5670A",
            "\
            37E6F9B266A6079AF25349E1B43A35C37FBD950FB3400A28119C80AEAF852FCE\
            8AA4BC0F02EC5527FC15B8D61EEA7C8819F80C0D4F5FBADF62317CF2C7E09216\
            AE8639899B55BD96059BF955BFD34875FE3AFFB038821489CB338341265D9DB8\
            9B5A6784936B7EE9CFFB694044D8424103A417C9D4CAE0DA09E7C945B6DC7E35\
            6D3EA39F0DE60E0AB7F386825CDE9F3F017F5BB5CD0417C09589E315455A9DDF\
            09190BCB38BD86CF321FCCCABF91A5B0E790F716940D8A1BA4C49CAC51107718\
            3BAE3CEF2EB49DC8E2167E7A95643C515641D419F3E0E123D8DFACB27CBD2357\
            9FF0F220F8825BE10474436938056C9AC212FE05F2644E500B0A4423134A76FA",
            "7605FD378278F8BF1ABE054C510E338B6197A2D7A7EACEA84D2A28E64155B510",
            "68E1C9EC4C7B57AC9B8D9C461E5FF1C0A9D14B0E808CDFEE3958A3BD1D8E4B00",
            false,
        ),
        (
            "\
            3C5BD2D888989BA73AACCF0A2DCEC21479093831B1CC2F888D5F483AD2AE9B57\
            CEF77D98AF343FBE68AE20A90239BC41ED267A9F7897B200C92EA0BB54B74BCE\
            942AA1DF27ACD09EFDEE04C9587B0C94CCFD8330C113B2034E7F0270FF17EAAE\
            9021A6463D402DCF1E1991EB3321941243D4BC1D0AFBE94BA608B816C3862ACF",
            "\
            18EBEF7780F789186E1B2E53F726C6A55E332396CB251B30CCD7C7361C82E021\
            BEE62BD2BB9C407DC7D25A6FFA8EBDCE41F49E3C779BE5E0F23DFE5EBE3F7337\
            AFED2BC0F9788B33CAA9C5C5ED79EE81804E95350BD9DDC96A817B4E0431FC10\
            FB0D90A16624015490868EA1EEBC093AA20DF0323CEED8B743EE874E39006F67\
            722789DC0BCE6AEE42A1C4F06265077B8D1EE68757E75F0088ABB2822A931567\
            5B68E4F3415FEE9856336A83D289AC30510BEDD9AA12586184CAAA64677AA131\
            B240F04E0A36619A230CCD275D1946BDDF8F116312EEDE37356023EBB8F16860\
            F608C10E8C180FED37FA292CAC2C4F57710486E934A4CB4F82868B137AB92CA8",
            "36B89BCFD5065DE95E46C63933719B1EA2646A4B2CC70FAF4BBE0C3B0EE9C118",
            "10A5C796245F32D62C8538589264501355B71C6DF67CEA07B5E9D986D67AA8C1",
            true,
        ),
    ];

    /// Reads the parameters of one group into working values.
    fn parameters(p: &str, q: &str, g: &str, y: &str) -> (BigUint, BigUint, BigUint, BigUint) {
        (
            BigUint::from_be_bytes(&hex(p)),
            BigUint::from_be_bytes(&hex(q)),
            BigUint::from_be_bytes(&hex(g)),
            BigUint::from_be_bytes(&hex(y)),
        )
    }

    /// The signature XML Signature §6.4.1 prints as its worked example is
    /// exactly `r ‖ s` of the FIPS 186-2 Appendix 5 example, base64 encoded — so
    /// checking the base64 checks this crate's understanding of the encoding as
    /// well as of the arithmetic.
    #[test]
    fn the_xml_signature_worked_example_verifies() {
        let (p, q, g, y) = parameters(EXAMPLE_P, EXAMPLE_Q, EXAMPLE_G, EXAMPLE_Y);
        let mut signature = hex(EXAMPLE_R);
        signature.extend_from_slice(&hex(EXAMPLE_S));
        assert_eq!(
            oxixml_text::base64::base64_encode(&signature),
            "i6watmQQQ1y3GB+VsWq5fJKzQcBB4jRfH1bfJFj0JtFVtLotttzYyA=="
        );

        let key = PublicParameters {
            p: &p,
            q: &q,
            g: &g,
            y: &y,
        };
        assert!(verify(key, DigestAlgorithm::Sha1, b"abc", &signature, 20));
        // A different message, a flipped bit and a swapped pair must all fail.
        assert!(!verify(key, DigestAlgorithm::Sha1, b"abd", &signature, 20));
        let mut tampered = signature.clone();
        if let Some(last) = tampered.last_mut() {
            *last ^= 0x01;
        }
        assert!(!verify(key, DigestAlgorithm::Sha1, b"abc", &tampered, 20));
        let mut swapped = hex(EXAMPLE_S);
        swapped.extend_from_slice(&hex(EXAMPLE_R));
        assert!(!verify(key, DigestAlgorithm::Sha1, b"abc", &swapped, 20));
    }

    /// A signature this crate produces for the same key must verify — and, being
    /// deterministic, must come out identical on a second run.
    #[test]
    fn sign_and_verify_round_trip() {
        let (p, q, g, y) = parameters(EXAMPLE_P, EXAMPLE_Q, EXAMPLE_G, EXAMPLE_Y);
        let x = BigUint::from_be_bytes(&hex(EXAMPLE_X));
        let key = PublicParameters {
            p: &p,
            q: &q,
            g: &g,
            y: &y,
        };
        for algorithm in [DigestAlgorithm::Sha1, DigestAlgorithm::Sha256] {
            let signature = sign(&p, &q, &g, &x, algorithm, b"SignedInfo octets", 20)
                .expect("the private value is in range");
            assert_eq!(signature.len(), 40);
            assert!(verify(key, algorithm, b"SignedInfo octets", &signature, 20));
            assert!(!verify(key, algorithm, b"other octets", &signature, 20));
            assert_eq!(
                sign(&p, &q, &g, &x, algorithm, b"SignedInfo octets", 20),
                Some(signature)
            );
        }
    }

    /// The published key satisfies `y = g^x mod p`, which is the only thing that
    /// ties the private value in the example to its public half.
    #[test]
    fn the_example_public_value_matches_its_private_one() {
        let (p, _q, g, y) = parameters(EXAMPLE_P, EXAMPLE_Q, EXAMPLE_G, EXAMPLE_Y);
        let x = BigUint::from_be_bytes(&hex(EXAMPLE_X));
        assert_eq!(g.modpow(&x, &p), Some(y));
    }

    /// The NIST CAVP `SigVer` vectors for FIPS 186-3, which include deliberately
    /// corrupted signatures: every `Result = F` row must be rejected, and for a
    /// reason the verifier reaches on its own rather than by luck.
    #[test]
    fn cavp_sig_ver_vectors() {
        /// One row of a CAVP `SigVer` file: `(message, y, r, s, expected)`.
        type Case = (&'static str, &'static str, &'static str, &'static str, bool);

        /// One `[mod = …]` group: its domain parameters, the digest it was
        /// signed with, the signature width and its rows.
        struct Group {
            p: &'static str,
            q: &'static str,
            g: &'static str,
            digest: DigestAlgorithm,
            half_width: usize,
            cases: &'static [Case],
        }

        let groups = [
            Group {
                p: CAVP_1024_SHA1_P,
                q: CAVP_1024_SHA1_Q,
                g: CAVP_1024_SHA1_G,
                digest: DigestAlgorithm::Sha1,
                half_width: 20,
                cases: CAVP_1024_SHA1_CASES.as_slice(),
            },
            Group {
                p: CAVP_2048_SHA256_P,
                q: CAVP_2048_SHA256_Q,
                g: CAVP_2048_SHA256_G,
                digest: DigestAlgorithm::Sha256,
                half_width: 32,
                cases: CAVP_2048_SHA256_CASES.as_slice(),
            },
        ];
        for group in groups {
            for (index, (message, y, r, s, expected)) in group.cases.iter().enumerate() {
                let (p, q, g, y) = parameters(group.p, group.q, group.g, y);
                let key = PublicParameters {
                    p: &p,
                    q: &q,
                    g: &g,
                    y: &y,
                };
                let mut signature = hex(r);
                signature.extend_from_slice(&hex(s));
                assert_eq!(
                    verify(
                        key,
                        group.digest,
                        &hex(message),
                        &signature,
                        group.half_width
                    ),
                    *expected,
                    "{} case {index}",
                    group.digest
                );
            }
        }
    }

    #[test]
    fn a_signature_of_the_wrong_length_is_rejected() {
        let (p, q, g, y) = parameters(EXAMPLE_P, EXAMPLE_Q, EXAMPLE_G, EXAMPLE_Y);
        let key = PublicParameters {
            p: &p,
            q: &q,
            g: &g,
            y: &y,
        };
        let mut signature = hex(EXAMPLE_R);
        signature.extend_from_slice(&hex(EXAMPLE_S));

        // The `#dsa-sha256` width applied to a `#dsa-sha1` signature value.
        assert!(!verify(key, DigestAlgorithm::Sha1, b"abc", &signature, 32));
        // Truncated, and padded.
        assert!(!verify(
            key,
            DigestAlgorithm::Sha1,
            b"abc",
            signature.get(..39).unwrap_or_default(),
            20
        ));
        let mut padded = signature.clone();
        padded.push(0);
        assert!(!verify(key, DigestAlgorithm::Sha1, b"abc", &padded, 20));
    }

    #[test]
    fn a_component_outside_the_subgroup_order_is_rejected() {
        let (p, q, g, y) = parameters(EXAMPLE_P, EXAMPLE_Q, EXAMPLE_G, EXAMPLE_Y);
        let key = PublicParameters {
            p: &p,
            q: &q,
            g: &g,
            y: &y,
        };
        let order = q.to_be_bytes_padded(20).expect("q is 160 bits");

        // r = 0
        let mut zero_r = vec![0_u8; 20];
        zero_r.extend_from_slice(&hex(EXAMPLE_S));
        assert!(!verify(key, DigestAlgorithm::Sha1, b"abc", &zero_r, 20));

        // s = q
        let mut order_s = hex(EXAMPLE_R);
        order_s.extend_from_slice(&order);
        assert!(!verify(key, DigestAlgorithm::Sha1, b"abc", &order_s, 20));
    }

    #[test]
    fn a_private_value_outside_the_subgroup_order_is_refused() {
        let (p, q, g, _y) = parameters(EXAMPLE_P, EXAMPLE_Q, EXAMPLE_G, EXAMPLE_Y);
        assert!(
            sign(
                &p,
                &q,
                &g,
                &BigUint::zero(),
                DigestAlgorithm::Sha1,
                b"abc",
                20
            )
            .is_none()
        );
        assert!(sign(&p, &q, &g, &q, DigestAlgorithm::Sha1, b"abc", 20).is_none());
    }
}
