use std::collections::HashMap;

use crate::Transform;

/// A name -> plug-in directory the CLI uses to dispatch `chord <name>`.
///
/// The registry does **not** route or chain transforms — composition is the
/// shell pipe's job. It only stores plug-ins and lists them (in registration
/// order) for dispatch and for `chord ls`.
#[derive(Default)]
pub struct Registry {
    by_name: HashMap<String, Box<dyn Transform>>,
    order: Vec<String>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a plug-in. Panics on a duplicate name, which is a wiring bug in
    /// the composition root, not a runtime condition.
    pub fn register(&mut self, t: Box<dyn Transform>) {
        let name = t.name().to_string();
        if self.by_name.contains_key(&name) {
            panic!("chord: duplicate transform name: {name}");
        }
        self.order.push(name.clone());
        self.by_name.insert(name, t);
    }

    /// Look up a plug-in by name.
    pub fn get(&self, name: &str) -> Option<&dyn Transform> {
        self.by_name.get(name).map(Box::as_ref)
    }

    /// Iterate plug-ins in registration order.
    pub fn all(&self) -> impl Iterator<Item = &dyn Transform> {
        self.order.iter().map(move |n| self.by_name[n].as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Kind, Options, Result, Unary};
    use std::io::{Read, Write};

    struct Noop(&'static str);
    impl Unary for Noop {
        fn name(&self) -> &str {
            self.0
        }
        fn from(&self) -> Kind {
            Kind::Text
        }
        fn to(&self) -> Kind {
            Kind::Text
        }
        fn describe(&self) -> &str {
            "noop"
        }
        fn apply(&self, _: &mut dyn Read, _: &mut dyn Write, _: &Options) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn register_get_and_order() {
        let mut r = Registry::new();
        r.register(Box::new(Noop("a")));
        r.register(Box::new(Noop("b")));

        assert!(r.get("a").is_some());
        assert!(r.get("missing").is_none());

        let names: Vec<_> = r.all().map(|t| t.name().to_string()).collect();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    #[should_panic(expected = "duplicate transform name")]
    fn duplicate_name_panics() {
        let mut r = Registry::new();
        r.register(Box::new(Noop("dup")));
        r.register(Box::new(Noop("dup")));
    }
}
