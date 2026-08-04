# SeqTrainer architecture

## Layering

1. **Domain core (framework-neutral)**
   - `seqtrainer.clients`
   - `seqtrainer.sparql`
   - `seqtrainer.data`
   - `seqtrainer.transforms`
   - `seqtrainer.models`

2. **Framework adapters (optional deps)**
   - `seqtrainer.keras`
   - `seqtrainer.torch`

3. **Graph utilities**
   - `seqtrainer.graph`

4. **Task applications**
   - `seqtrainer.applications`

5. **Delivery surfaces**
   - `seqtrainer.cli`

## Extension strategy

- Keep SBOL/SynBioHub semantics in domain modules, not in framework wrappers.
- Add new tasks as application blueprints that reference:
  - a dataset recipe,
  - transforms,
  - backbone/head choice,
  - adapter path.
- Use lazy imports in framework modules so base install remains lightweight.

## Experimental code policy

Prototype scripts and notebooks remain available, but stable APIs should be surfaced through modules under `seqtrainer/*`.

## Experimental backbones

- Experimental sequence backbones and prototype training loops should live under `seqtrainer.torch`, while stable framework-neutral metadata and registries remain under `seqtrainer.models`.
