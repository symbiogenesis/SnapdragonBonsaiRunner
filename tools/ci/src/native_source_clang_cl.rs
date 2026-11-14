//! Original clang-cl argument arrays; never translate a shell command or GNU flags.

use super::{Unit, path_key};
use crate::Result;
use std::collections::BTreeSet;

fn output_option<'a>(argument: &'a str, prefix: &str) -> Option<&'a str> {
    argument
        .strip_prefix('/')
        .or_else(|| argument.strip_prefix('-'))?
        .strip_prefix(prefix)
}

fn record<'a>(
    selected: &mut BTreeSet<&'a str>,
    name: &'a str,
    value: &str,
    expected: &str,
) -> Result<()> {
    // Preserve the original argv, but compare ordinary absolute file aliases by
    // identity, as the compilation-database and dependency verifiers already do.
    if value.is_empty() || !selected.insert(name) || path_key(value)? != path_key(expected)? {
        return Err("Clang-cl capture has a duplicate or changed output identity.".into());
    }
    Ok(())
}

pub(super) fn configuration(unit: &Unit, preprocess: bool) -> Result<Vec<&str>> {
    let arguments = if preprocess {
        &unit.preprocessing_arguments
    } else {
        &unit.arguments
    };
    let mut selected = BTreeSet::new();
    let mut common = Vec::new();
    let mut index = 1;
    while index < arguments.len() {
        let argument = arguments[index].as_str();
        if argument == "--" {
            // clang-cl's explicit end-of-options delimiter may precede only the
            // single reviewed translation unit. Keep it in the configuration.
            if index + 2 != arguments.len() || arguments[index + 1] != unit.source {
                return Err("Clang-cl capture has additional or changed source inputs.".into());
            }
            common.extend([argument, arguments[index + 1].as_str()]);
            break;
        }
        if matches!(argument, "/c" | "-c" | "/P" | "-P") {
            let action = if argument.ends_with('c') {
                "compile"
            } else {
                "preprocess"
            };
            if !selected.insert(action) {
                return Err("Clang-cl capture repeats an action option.".into());
            }
        } else if let Some(mut value) = output_option(argument, "Fo") {
            if value.is_empty() {
                index += 1;
                value = arguments
                    .get(index)
                    .ok_or("Missing clang-cl object path.")?;
            }
            if preprocess {
                return Err("Clang-cl preprocessing cannot select an object output.".into());
            }
            record(&mut selected, "object", value, &unit.object.path)?;
        } else if let Some(mut value) = output_option(argument, "Fi") {
            if value.is_empty() {
                index += 1;
                value = arguments
                    .get(index)
                    .ok_or("Missing clang-cl preprocessing path.")?;
            }
            if !preprocess {
                return Err("Clang-cl compilation cannot select a preprocessing output.".into());
            }
            record(
                &mut selected,
                "preprocessed",
                value,
                &unit.preprocessed.path,
            )?;
        } else if let Some(forwarded) = output_option(argument, "clang:") {
            if preprocess
                && (forwarded == "-MD"
                    || forwarded.starts_with("-MF")
                    || forwarded.starts_with("-MT"))
            {
                return Err("Clang-cl preprocessing cannot change forwarded options.".into());
            }
            if forwarded == "-MD" {
                if !selected.insert("dependencies") {
                    return Err("Clang-cl capture repeats dependency generation.".into());
                }
            } else if let Some(value) = forwarded.strip_prefix("-MF") {
                record(&mut selected, "depfile", value, &unit.dependencies.path)?;
            } else if let Some(value) = forwarded.strip_prefix("-MT") {
                record(&mut selected, "target", value, &unit.object.path)?;
            } else {
                // ISA and other common configuration is retained unchanged in
                // both invocations, not treated as a dependency-generation flag.
                if !common_forwarded(forwarded) {
                    return Err("Unsupported clang-cl forwarded capture option.".into());
                }
                common.push(argument);
            }
        } else {
            if argument != unit.source && !common_configuration(argument) {
                return Err("Unsupported clang-cl capture option.".into());
            }
            common.push(argument);
        }
        index += 1;
    }
    let required = if preprocess {
        BTreeSet::from(["preprocess", "preprocessed"])
    } else {
        BTreeSet::from(["compile", "object", "dependencies", "depfile", "target"])
    };
    if selected != required
        || common
            .iter()
            .filter(|argument| **argument == unit.source)
            .count()
            != 1
    {
        return Err(
            "Clang-cl capture needs explicit actions, outputs and complete dependencies.".into(),
        );
    }
    Ok(common)
}

fn common_forwarded(argument: &str) -> bool {
    ["-march=", "-mcpu=", "-mtune=", "-ffp-contract="]
        .iter()
        .any(|prefix| {
            argument
                .strip_prefix(prefix)
                .is_some_and(|value| !value.is_empty())
        })
}

fn common_configuration(argument: &str) -> bool {
    if argument.starts_with("--target=") {
        return argument.len() > "--target=".len();
    }
    let Some(option) = argument
        .strip_prefix('/')
        .or_else(|| argument.strip_prefix('-'))
    else {
        return false;
    };
    // Joined configuration only: an unparsed positional value could otherwise
    // hide a second source or an action/output override. Spellings are
    // case-sensitive in clang-cl (notably /FI versus /Fi, /TP versus /Tp).
    matches!(
        option,
        "nologo"
            | "TP"
            | "TC"
            | "MD"
            | "MDd"
            | "MT"
            | "MTd"
            | "O0"
            | "O1"
            | "O2"
            | "Od"
            | "Os"
            | "Ot"
            | "Ox"
            | "Ob0"
            | "Ob1"
            | "Ob2"
            | "Ob3"
            | "W0"
            | "W1"
            | "W2"
            | "W3"
            | "W4"
            | "Wall"
            | "WX"
            | "WX-"
            | "utf-8"
            | "GR"
            | "GR-"
            | "GS"
            | "GS-"
            | "Z7"
            | "X"
            | "permissive"
            | "permissive-"
    ) || [
        "D",
        "U",
        "I",
        "FI",
        "imsvc",
        "external:I",
        "external:W",
        "EH",
        "std:",
        "std=",
        "fp:",
        "arch:",
        "Zc:",
        "wd",
        "we",
        "wo",
        "volatile:",
        "char:",
    ]
    .iter()
    .any(|prefix| {
        option
            .strip_prefix(prefix)
            .is_some_and(|value| !value.is_empty())
    })
}
