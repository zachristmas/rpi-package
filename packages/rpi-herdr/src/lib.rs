#![forbid(unsafe_code)]
//! Reports rpi's lifecycle to Herdr so a pane running rpi shows up as an agent.
//!
//! Follows Herdr's "Add Herdr support to your agent" contract: state reports,
//! the resume command and a release on exit, all through
//! `$HERDR_BIN_PATH pane report-agent|release-agent`. Reporting never blocks
//! rpi: reports go to one background thread, newer ones replace older ones that
//! have not been sent yet, and every failure is ignored.
use rpi_plugin_sdk::EventTag;
use std::{
    io::IsTerminal,
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc, Mutex, OnceLock,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const SOURCE: &str = "rpi-herdr";
const AGENT: &str = "rpi";
const SEND_TIMEOUT: Duration = Duration::from_secs(2);
const RELEASE_WAIT: Duration = Duration::from_secs(3);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Idle,
    Working,
    Blocked,
}

impl State {
    fn as_str(self) -> &'static str {
        match self {
            State::Idle => "idle",
            State::Working => "working",
            State::Blocked => "blocked",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Action {
    Nothing,
    Report(State),
    Release,
}

/// What rpi is doing, derived from lifecycle events. Pure so it can be tested
/// without Herdr.
#[derive(Default)]
struct Tracker {
    interactive: bool,
    active: bool,
    prompts: u32,
    last: Option<State>,
}

impl Tracker {
    fn state(&self) -> State {
        match (self.active, self.prompts) {
            (false, _) => State::Idle,
            (true, 0) => State::Working,
            (true, _) => State::Blocked,
        }
    }

    /// `terminal` says whether stdin and stdout are both terminals. Together
    /// with `BeforeTuiStart` that identifies an interactive session; headless
    /// RPC and print runs also reach other events but are never reported.
    fn apply(&mut self, tag: EventTag, terminal: bool) -> Action {
        let mut force = false;
        match tag {
            EventTag::BeforeTuiStart => {
                self.interactive = terminal;
                self.active = false;
                self.prompts = 0;
                force = true;
            }
            // A new or switched session refreshes the resume command.
            EventTag::SessionStart => force = true,
            EventTag::AgentStart => self.active = true,
            EventTag::AgentEnd | EventTag::AgentSettled => {
                self.active = false;
                self.prompts = 0;
            }
            EventTag::UiPromptStart => self.prompts += 1,
            EventTag::UiPromptEnd => self.prompts = self.prompts.saturating_sub(1),
            EventTag::SessionShutdown => {
                let was = self.interactive;
                self.interactive = false;
                self.last = None;
                return if was {
                    Action::Release
                } else {
                    Action::Nothing
                };
            }
            _ => return Action::Nothing,
        }
        if !self.interactive {
            return Action::Nothing;
        }
        let state = self.state();
        if !force && self.last == Some(state) {
            return Action::Nothing;
        }
        self.last = Some(state);
        Action::Report(state)
    }
}

struct Target {
    bin: String,
    pane: String,
}

fn target() -> Option<Target> {
    let var = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
    if var("HERDR_ENV")? != "1" {
        return None;
    }
    Some(Target {
        bin: var("HERDR_BIN_PATH")?,
        pane: var("HERDR_PANE_ID")?,
    })
}

/// Strictly increasing across reports, sessions and restarts.
fn next_seq() -> u64 {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_micros() as u64);
    let mut last = LAST.load(Ordering::Relaxed);
    loop {
        let next = now.max(last + 1);
        match LAST.compare_exchange(last, next, Ordering::SeqCst, Ordering::Relaxed) {
            Ok(_) => return next,
            Err(seen) => last = seen,
        }
    }
}

fn state_args(pane: &str, state: State, seq: u64, session: Option<&str>) -> Vec<String> {
    let mut args = vec![
        "pane".to_string(),
        "report-agent".into(),
        pane.into(),
        "--source".into(),
        SOURCE.into(),
        "--agent".into(),
        AGENT.into(),
        "--state".into(),
        state.as_str().into(),
        "--seq".into(),
        seq.to_string(),
    ];
    if state == State::Blocked {
        args.extend(["--message".into(), "waiting for input".into()]);
    }
    if let Some(id) = session {
        // The resume command: the first word is a plain command name on PATH
        // and nothing in it contains an apostrophe or control character.
        args.extend(["--agent-session-id".into(), id.into()]);
        args.extend(["--".into(), AGENT.into(), "--session".into(), id.into()]);
    }
    args
}

fn release_args(pane: &str, seq: u64) -> Vec<String> {
    vec![
        "pane".to_string(),
        "release-agent".into(),
        pane.into(),
        "--source".into(),
        SOURCE.into(),
        "--agent".into(),
        AGENT.into(),
        "--seq".into(),
        seq.to_string(),
    ]
}

/// The id of the session this process serves; the host keeps it current.
fn session_id() -> Option<String> {
    std::env::var(rpi_plugin_sdk::SESSION_ID_ENV)
        .ok()
        .filter(|id| {
            !id.is_empty()
                && id.len() <= 128
                && id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        })
}

/// Runs the Herdr CLI with a hard timeout. Output and failures are discarded.
fn run(bin: &str, args: &[String]) {
    let Ok(mut child) = Command::new(bin)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return;
    };
    let deadline = Instant::now() + SEND_TIMEOUT;
    while Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(_)) | Err(_) => return,
            Ok(None) => thread::sleep(Duration::from_millis(15)),
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

enum Message {
    State(State, u64, Option<String>),
    Release(u64, mpsc::Sender<()>),
}

fn worker(target: Target, rx: mpsc::Receiver<Message>) {
    while let Ok(mut message) = rx.recv() {
        // Only the latest state matters; drop older ones that queued up while
        // the previous request was running. A release is always delivered.
        if !matches!(message, Message::Release(..)) {
            while let Ok(newer) = rx.try_recv() {
                message = newer;
                if matches!(message, Message::Release(..)) {
                    break;
                }
            }
        }
        match message {
            Message::State(state, seq, session) => run(
                &target.bin,
                &state_args(&target.pane, state, seq, session.as_deref()),
            ),
            Message::Release(seq, done) => {
                run(&target.bin, &release_args(&target.pane, seq));
                let _ = done.send(());
            }
        }
    }
}

static SENDER: OnceLock<mpsc::Sender<Message>> = OnceLock::new();
static TRACKER: Mutex<Tracker> = Mutex::new(Tracker {
    interactive: false,
    active: false,
    prompts: 0,
    last: None,
});

fn dispatch(action: Action) {
    let Some(sender) = SENDER.get() else {
        return;
    };
    match action {
        Action::Nothing => {}
        Action::Report(state) => {
            let _ = sender.send(Message::State(state, next_seq(), session_id()));
        }
        Action::Release => {
            let (done, wait) = mpsc::channel();
            if sender.send(Message::Release(next_seq(), done)).is_ok() {
                let _ = wait.recv_timeout(RELEASE_WAIT);
            }
        }
    }
}

rpi_plugin_sdk::event_handler! {
fn on_event(event, _user_data) {
    let terminal = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    let action = TRACKER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .apply(event.tag(), terminal);
    dispatch(action);
    0
}
}

rpi_plugin_sdk::export_plugin!(|api| {
    // Outside Herdr the extension does nothing at all.
    let Some(target) = target() else {
        return 0;
    };
    let Some(register) = api.register_event_handler else {
        return 1;
    };
    let (tx, rx) = mpsc::channel();
    if SENDER.set(tx).is_err()
        || thread::Builder::new()
            .name("rpi-herdr".into())
            .spawn(move || worker(target, rx))
            .is_err()
    {
        return 1;
    }
    for tag in [
        EventTag::BeforeTuiStart,
        EventTag::SessionStart,
        EventTag::SessionShutdown,
        EventTag::AgentStart,
        EventTag::AgentEnd,
        EventTag::AgentSettled,
        EventTag::UiPromptStart,
        EventTag::UiPromptEnd,
    ] {
        if register(tag, on_event, std::ptr::null_mut()) != 0 {
            return 1;
        }
    }
    0
});

#[cfg(test)]
mod tests {
    use super::*;

    fn report(state: State) -> Action {
        Action::Report(state)
    }

    #[test]
    fn follows_a_run_from_idle_to_working_to_blocked_and_back() {
        let mut t = Tracker::default();
        assert_eq!(t.apply(EventTag::BeforeTuiStart, true), report(State::Idle));
        assert_eq!(t.apply(EventTag::AgentStart, true), report(State::Working));
        assert_eq!(
            t.apply(EventTag::UiPromptStart, true),
            report(State::Blocked)
        );
        assert_eq!(t.apply(EventTag::UiPromptEnd, true), report(State::Working));
        assert_eq!(t.apply(EventTag::AgentEnd, true), report(State::Idle));
        // A settled event right after the end changes nothing.
        assert_eq!(t.apply(EventTag::AgentSettled, true), Action::Nothing);
    }

    #[test]
    fn a_selector_opened_while_idle_is_not_a_block() {
        let mut t = Tracker::default();
        t.apply(EventTag::BeforeTuiStart, true);
        assert_eq!(t.apply(EventTag::UiPromptStart, true), Action::Nothing);
        assert_eq!(t.apply(EventTag::UiPromptEnd, true), Action::Nothing);
        // A prompt left open when the run starts blocks it only once counted.
        t.apply(EventTag::UiPromptStart, true);
        assert_eq!(t.apply(EventTag::AgentStart, true), report(State::Blocked));
    }

    #[test]
    fn nested_prompts_unblock_only_when_all_have_closed() {
        let mut t = Tracker::default();
        t.apply(EventTag::BeforeTuiStart, true);
        t.apply(EventTag::AgentStart, true);
        assert_eq!(
            t.apply(EventTag::UiPromptStart, true),
            report(State::Blocked)
        );
        assert_eq!(t.apply(EventTag::UiPromptStart, true), Action::Nothing);
        assert_eq!(t.apply(EventTag::UiPromptEnd, true), Action::Nothing);
        assert_eq!(t.apply(EventTag::UiPromptEnd, true), report(State::Working));
        // An extra end never underflows.
        assert_eq!(t.apply(EventTag::UiPromptEnd, true), Action::Nothing);
    }

    #[test]
    fn headless_runs_never_report() {
        let mut t = Tracker::default();
        for tag in [
            EventTag::BeforeTuiStart,
            EventTag::SessionStart,
            EventTag::AgentStart,
            EventTag::AgentEnd,
            EventTag::SessionShutdown,
        ] {
            assert_eq!(t.apply(tag, false), Action::Nothing);
        }
        // Print mode never fires BeforeTuiStart, so a terminal alone is not enough.
        let mut t = Tracker::default();
        assert_eq!(t.apply(EventTag::SessionStart, true), Action::Nothing);
        assert_eq!(t.apply(EventTag::AgentStart, true), Action::Nothing);
    }

    #[test]
    fn session_changes_resend_the_state_and_shutdown_releases_once() {
        let mut t = Tracker::default();
        t.apply(EventTag::BeforeTuiStart, true);
        assert_eq!(t.apply(EventTag::SessionStart, true), report(State::Idle));
        assert_eq!(t.apply(EventTag::SessionShutdown, true), Action::Release);
        assert_eq!(t.apply(EventTag::SessionShutdown, true), Action::Nothing);
        assert_eq!(t.apply(EventTag::AgentStart, true), Action::Nothing);
    }

    #[test]
    fn sequence_numbers_only_increase() {
        let seqs: Vec<u64> = (0..1000).map(|_| next_seq()).collect();
        assert!(seqs.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn state_report_carries_the_resume_command() {
        let args = state_args("w1:p1", State::Blocked, 7, Some("abc-123"));
        assert_eq!(
            args.join(" "),
            "pane report-agent w1:p1 --source rpi-herdr --agent rpi --state blocked --seq 7 \
             --message waiting for input --agent-session-id abc-123 -- rpi --session abc-123"
        );
        let plain = state_args("w1:p1", State::Working, 8, None);
        assert!(!plain.contains(&"--".to_string()));
        assert_eq!(
            release_args("w1:p1", 9).join(" "),
            "pane release-agent w1:p1 --source rpi-herdr --agent rpi --seq 9"
        );
    }

    #[test]
    fn session_ids_with_unsafe_characters_are_dropped() {
        // The resume command may not contain apostrophes or control characters.
        std::env::set_var(rpi_plugin_sdk::SESSION_ID_ENV, "bad'id");
        assert_eq!(session_id(), None);
        std::env::set_var(rpi_plugin_sdk::SESSION_ID_ENV, "0199-abcd_1.x");
        assert_eq!(session_id().as_deref(), Some("0199-abcd_1.x"));
        std::env::remove_var(rpi_plugin_sdk::SESSION_ID_ENV);
    }
}
