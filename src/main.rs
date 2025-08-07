use std::ptr::NonNull;

use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle,
};
use wayland_client::{
    Connection, Dispatch, Proxy, QueueHandle, delegate_noop,
    protocol::{wl_compositor, wl_registry, wl_surface},
};
use wayland_protocols::xdg::shell::client::{
    xdg_surface,
    xdg_toplevel::{self, XdgToplevel},
    xdg_wm_base,
};
use wgpu::SurfaceTargetUnsafe;

// Application State
//
// The `AppState` struct holds all the application-level state,
// including Wayland objects, window configuration, and GPU context.
//
// ─────────────────────────────────────────────────────────────
//
// The core event dispatching logic is built around the `EventQueue`.
// Receiving and processing events is a two-step process:
//
//   1. Events are read from the Wayland socket and assigned to an `EventQueue`.
//   2. The queue then dispatches these events by calling the appropriate
//      `Dispatch::event()` implementation on the provided `State`.
//
// The design ensures that your application's state can be accessed and
// mutated directly during event handling — without needing synchronization.
// This helps reduce overhead and simplifies logic.
//
// ─────────────────────────────────────────────────────────────
//
// Creating an `xdg_surface` alone does NOT assign a role to the `wl_surface`.
// You must immediately assign a role-specific object, like `get_toplevel()` or `get_popup()`.
// A `wl_surface` can only have one role, and it must match the `xdg_surface`-based role.
//
// ─────────────────────────────────────────────────────────────
struct AppState {
    running: bool,
    //Wayland objects
    wl_surface: Option<wl_surface::WlSurface>,
    wm_base: Option<xdg_wm_base::XdgWmBase>,
    xdg_surface: Option<xdg_surface::XdgSurface>,
    xdg_toplevel: Option<xdg_toplevel::XdgToplevel>,

    //Window Config
    configured: bool,
    size: Option<WindowSize>,
    pending_resize: Option<WindowSize>,

    //GPU
    wgpu_state: Option<WgpuState>,
}

struct WindowSize {
    width: i32,
    height: i32,
}

impl Default for WindowSize {
    fn default() -> Self {
        Self {
            width: 320,
            height: 320,
        }
    }
}

struct WgpuState {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            running: true,
            wl_surface: None,
            wm_base: None,
            xdg_surface: None,
            xdg_toplevel: None,
            size: None,
            pending_resize: None,
            configured: false,
            wgpu_state: None,
        }
    }
}

impl AppState {
    fn init_xdg_surface(&mut self, queue_handle: &QueueHandle<AppState>) {
        // ─────────────────────────────────────────────────────────────
        // `xdg_wm_base`
        //
        // The `xdg_wm_base` global is exposed by the compositor to allow clients
        // to turn `wl_surface`s into "windows" in a desktop environment.
        //
        // It acts as the entry point for the XDG shell protocol and is responsible
        // for creating `xdg_surface` objects.
        // ─────────────────────────────────────────────────────────────
        let wm_base = self.wm_base.as_ref().expect(
            "wm_base is None - Make sure to bind xdg_wm_base before trying to create a xdg_surface",
        );

        // ─────────────────────────────────────────────────────────────
        // `wl_surface`
        //
        // The `wl_surface` is a low-level rectangle area that clients use to:
        //
        //   - Attach graphical content via `wl_buffer`s
        //   - Receive input events
        //   - Define local coordinate systems
        //
        // ─────────────────────────────────────────────────────────────
        let wl_surface = self.wl_surface.as_ref().expect("wl_surface is None - Create it via wl_compositor before attempting to create a xdg_surface");

        // ─────────────────────────────────────────────────────────────
        // `xdg_surface`
        //
        // The `xdg_surface` protocol is built on top of `wl_surface`.
        // It enables desktop-style window management (e.g, moving, resizing).
        //
        // To create an interactive window:
        //
        //   1. You must first create an `xdg_surface` from a `wl_surface`.
        //   2. Then, immediately assign it a role (e.g, `get_toplevel()`. `get_popup()`).
        //   3. Finally, perform an initial commit — This initial commit CANNOT have a buffer attached.
        //
        // ─────────────────────────────────────────────────────────────
        let xdg_surface = wm_base.get_xdg_surface(wl_surface, queue_handle, ());
        let xdg_toplevel = xdg_surface.get_toplevel(queue_handle, ());

        xdg_toplevel.set_title("receba".into());
        xdg_toplevel.set_app_id("EstamosAquiDaSilva.org".into());

        wl_surface.commit();

        self.xdg_surface = Some(xdg_surface);
        self.xdg_toplevel = Some(xdg_toplevel);
    }

    fn configure_wgpu(&self, width: i32, height: i32) {
        let wgpu_state = self
            .wgpu_state
            .as_ref()
            .expect("WgpuState is None - Make sure Wgpu is set up it before configuring");
        let wgpu_surface = &wgpu_state.surface;
        let device = &wgpu_state.device;
        let adapter = &wgpu_state.adapter;

        let capabilities = wgpu_surface.get_capabilities(adapter);

        let surface_configuration = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: capabilities.formats[0],
            width: width as u32,
            height: height as u32,
            present_mode: wgpu::PresentMode::Mailbox,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        wgpu_surface.configure(device, &surface_configuration);
    }

    fn init_wgpu(&mut self, connection: &Connection) {
        let instance_descriptor = wgpu::InstanceDescriptor::from_env_or_default();

        // ─────────────────────────────────────────────────────────────
        // `wgpu::Instance`
        //
        // The first object created in any WGPU app.
        // Used to create Adapters and Surfaces.
        // ─────────────────────────────────────────────────────────────
        let instance = wgpu::Instance::new(&instance_descriptor);

        let wayland_display_ptr = NonNull::new(connection.backend().display_ptr() as *mut _)
            .expect("Pointer to wl_display is null - Create a valid connection before attempting to init wgpu");
        let wayland_display_handle = WaylandDisplayHandle::new(wayland_display_ptr);
        let raw_display_handle = RawDisplayHandle::Wayland(wayland_display_handle);

        let wl_surface = self.wl_surface.as_ref().expect(
            "wl_surface is None - Create it via wl_compositor before attempting to init wgpu",
        );
        let wayland_surface_ptr = NonNull::new(wl_surface.id().as_ptr() as *mut _).unwrap();
        let wayland_window_handle = WaylandWindowHandle::new(wayland_surface_ptr);
        let raw_window_handle = RawWindowHandle::Wayland(wayland_window_handle);

        // ─────────────────────────────────────────────────────────────
        // `wgpu::Surface`
        //
        // The surface is the GPU draw target linked to a native window.
        //
        // Creating it requires raw pointers to:
        //   - The display (`wl_display`)
        //   - The window (`wl_surface`)
        //
        // These are passed through `raw-window-handle`, a cross-platform abstraction
        // that lets WGPU target Wayland, X11, Windows, etc.
        //
        // This block is marked `unsafe` because we're asserting the validity of
        // raw pointers. If they're null or misused, unknown behavior will occur.
        // ─────────────────────────────────────────────────────────────
        let wgpu_surface = unsafe {
            let surface_target = SurfaceTargetUnsafe::RawHandle {
                raw_display_handle,
                raw_window_handle,
            };

            instance.create_surface_unsafe(surface_target).unwrap()
        };

        // ─────────────────────────────────────────────────────────────
        // GPU Adapter Selection
        //
        // An Adapter represents a physical or virtual GPU.
        //
        // It exposes:
        //   - Hardware info (name, limits, features)
        //   - Backend compatibility (Vulkan, Metal, etc)
        //   - Methods to request a Device + Queue
        //
        // We pass the surface as a compatibility hint since not all adapters
        // support all surface types.
        // ─────────────────────────────────────────────────────────────
        let adapter_options = wgpu::RequestAdapterOptions {
            compatible_surface: Some(&wgpu_surface),
            ..Default::default()
        };
        let adapter = pollster::block_on(instance.request_adapter(&adapter_options))
            .expect("Cannot request adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default()))
            .expect("Cannot request device");

        let wgpu_state = WgpuState {
            surface: wgpu_surface,
            queue,
            device,
            adapter,
        };

        self.wgpu_state = Some(wgpu_state);
    }
}

fn draw(app_state: &AppState) {
    let wgpu_state = app_state
        .wgpu_state
        .as_ref()
        .expect("WgpuState is None - Make sure Wgpu is set up before drawing");

    let frame = wgpu_state
        .surface
        .get_current_texture()
        .expect("Failed to acquire next swapchain texture");

    let view = frame.texture.create_view(&Default::default());

    let mut encoder = wgpu_state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("draw_encoder"),
        });

    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("clear_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    wgpu_state.queue.submit(Some(encoder.finish()));
    frame.present();
}

// ─────────────────────────────────────────────────────────────
// Registry Binding (Wayland)
//
// The registry exposes all global objects (protocols/interfaces)
// that the compositor supports (e.g, `wl_compositor`, `wl_shm`).
//
// When each global is advertised, we bind the ones we need.
// Binding gives us a client-side handle to use those globals.
//
// Each bound interface needs an associated `Dispatch<O, _>` impl
// to handle its incoming events.
//
// `O` being the Wayland object that needs event processing
// (e.g, `wl_surface`, `xdg_wm_base`).
// ─────────────────────────────────────────────────────────────
impl Dispatch<wl_registry::WlRegistry, ()> for AppState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        queue_handle: &QueueHandle<AppState>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            match &interface[..] {
                "wl_compositor" => {
                    // ─────────────────────────────────────────────────────────────
                    // `wl_compositor`
                    //
                    // The compositor is responsible for creating the displayable
                    // output of multiple surfaces.
                    //
                    // It exposes functions to create `wl_surface`s and `wl_region`s
                    // ─────────────────────────────────────────────────────────────
                    let compositor = registry.bind::<wl_compositor::WlCompositor, _, _>(
                        name,
                        version,
                        queue_handle,
                        (),
                    );

                    let surface = compositor.create_surface(queue_handle, ());
                    state.wl_surface = Some(surface);
                }
                "xdg_wm_base" => {
                    // ─────────────────────────────────────────────────────────────
                    // `xdg_wm_base`
                    //
                    // Entry point for desktop-style windows and their features (drag, resize, maximize, etc.).
                    //
                    // It allows a `wl_surface` to become an `xdg_surface` and take on
                    // a role such as `xdg_toplevel` (main window) or `xdg_popup` (popup menu).
                    //
                    // Required for normal windows in desktop environments.
                    // ─────────────────────────────────────────────────────────────
                    let wm_base = registry.bind::<xdg_wm_base::XdgWmBase, _, _>(
                        name,
                        version,
                        queue_handle,
                        (),
                    );

                    state.wm_base = Some(wm_base);

                    if state.wl_surface.is_some() && state.xdg_surface.is_none() {
                        state.init_xdg_surface(queue_handle);
                    }
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<xdg_wm_base::XdgWmBase, ()> for AppState {
    fn event(
        _: &mut Self,
        wm_base: &xdg_wm_base::XdgWmBase,
        event: xdg_wm_base::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<AppState>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            wm_base.pong(serial);
        }
    }
}

impl Dispatch<xdg_surface::XdgSurface, ()> for AppState {
    fn event(
        state: &mut Self,
        surface_xdg: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<AppState>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            surface_xdg.ack_configure(serial);

            if let Some(size) = state.pending_resize.take() {
                state.configure_wgpu(size.width, size.height);
                state.size = Some(size);
            }

            state.configured = true;
        }
    }
}

impl Dispatch<xdg_toplevel::XdgToplevel, ()> for AppState {
    fn event(
        state: &mut Self,
        _: &XdgToplevel,
        event: xdg_toplevel::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<AppState>,
    ) {
        match event {
            xdg_toplevel::Event::Close => {
                state.running = false;
            }
            xdg_toplevel::Event::Configure { width, height, .. } => {
                state.pending_resize = Some(if (width, height) == (0, 0) {
                    // ─────────────────────────────────────────────────────────────
                    // Resize Behavior
                    //
                    // If the compositor sends a width/height of 0,
                    // it means the client is free to pick its own window size.
                    //
                    // This often happens on initial configuration or during certain resizes.
                    // ─────────────────────────────────────────────────────────────
                    WindowSize::default()
                } else {
                    WindowSize { width, height }
                });
            }
            _ => {}
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Ignored Protocols
//
// These protocol events (`wl_surface`, `wl_compositor`) are not handled
// in this app because we don't need their event streams.
//
// We delegate them to `noop`, satisfying the Dispatch requirement.
// ─────────────────────────────────────────────────────────────
delegate_noop!(AppState: ignore wl_compositor::WlCompositor);
delegate_noop!(AppState: ignore wl_surface::WlSurface);

fn main() {
    // ─────────────────────────────────────────────────────────────
    // Logging
    //
    // wgpu logs detailed errors through the `log` crate, but will panic
    // with a vague message if `env_logger::init()` isn't called.
    //
    // Without this, errors can fail silently and be nearly impossible to debug.
    // ─────────────────────────────────────────────────────────────
    env_logger::init();

    // ─────────────────────────────────────────────────────────────
    // Attempts to connect to the compositor based on environment config.
    // ─────────────────────────────────────────────────────────────
    let connection = Connection::connect_to_env().expect("Could not connect to a compositor.");

    // ─────────────────────────────────────────────────────────────
    // `wl_display`
    //
    // Root object in any Wayland program.
    // All Wayland objects are created from the display.
    // ─────────────────────────────────────────────────────────────
    let display = connection.display();

    // ─────────────────────────────────────────────────────────────
    // Event Queue
    //
    // Event queues process all incoming Wayland events.
    //
    // The handle is required to associate newly created objects
    // (like the registry) with this queue.
    // ─────────────────────────────────────────────────────────────
    let mut event_queue = connection.new_event_queue();
    let queue_handle = event_queue.handle();

    // ─────────────────────────────────────────────────────────────
    // `wl_registry`
    //
    // Used to list and bind to compositor-exposed globals.
    //
    // When created, it emits a event for each available interface.
    // Globals can appear or disappear dynamically (e.g, on hotplug or reconfig).
    //
    // Binding is done using the event queue handle, giving the client a local object
    // to send requests and receive events.
    //
    // The `Dispatch<wl_registry, _>` impl for AppState handles each global as it's announced.
    // ─────────────────────────────────────────────────────────────
    display.get_registry(&queue_handle, ());

    // Create our Application State.
    let mut app_state = AppState::default();

    // ─────────────────────────────────────────────────────────────
    // Initial Dispatch
    //
    // Blocking dispatch is used here to process registry events and
    // bind globals before initializing WGPU.
    //
    // This is a one-time setup dispatch.
    // ─────────────────────────────────────────────────────────────
    event_queue.blocking_dispatch(&mut app_state).unwrap();

    app_state.init_wgpu(&connection);

    //Application loop
    while app_state.running {
        // ─────────────────────────────────────────────────────────────
        // Application Loop
        //
        // Blocks on new Wayland events, processes them, and draws the frame.
        //
        // `blocking_dispatch()` will:
        //   - Flush the connection
        //   - Wait for new events from the compositor
        //   - Dispatch them to AppState
        // ─────────────────────────────────────────────────────────────
        event_queue.blocking_dispatch(&mut app_state).unwrap();

        // ─────────────────────────────────────────────────────────────
        // Rendering
        //
        // Once the surface has been configured by the compositor,
        // we proceed to draw to it.
        // ─────────────────────────────────────────────────────────────
        if app_state.configured {
            draw(&app_state);
        }
    }
}
