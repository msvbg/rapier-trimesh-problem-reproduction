mod level;

use bytemuck::{Pod, Zeroable};
use level::Level;
use lyon::{
    lyon_tessellation::{FillAttributes, FillVertexConstructor},
    math::Point,
};
use nalgebra::Vector2;
use nalgebra::Vector3;
use rapier2d::{data::arena::Index, dynamics::RigidBodyBuilder, geometry::Collider};
use rapier2d::{
    dynamics::BodyStatus, dynamics::IntegrationParameters, dynamics::JointSet,
    dynamics::RigidBodySet, geometry::Ball, geometry::BroadPhase, geometry::ColliderBuilder,
    geometry::ColliderSet, geometry::NarrowPhase, geometry::Shape, pipeline::PhysicsPipeline,
};
use std::{error::Error, time::Duration, time::Instant};
use tracing::info;
use tracing_subscriber::EnvFilter;
use wgpu::{
    util::DeviceExt, BindGroup, Device, Queue, RenderPipeline, Surface, SwapChain,
    SwapChainDescriptor,
};
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, VirtualKeyCode};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[derive(Default, Clone, Copy, Debug)]
pub struct Player;

pub struct Body {
    pub rigid_body_handle: Index,
    pub collider_handles: Vec<Index>,
}

impl Body {
    pub fn new(
        status: BodyStatus,
        colliders: Vec<Collider>,
        position: Vector3<f32>,
        body_set: &mut RigidBodySet,
        collider_set: &mut ColliderSet,
    ) -> Self {
        let rigid_body = RigidBodyBuilder::new(status)
            .translation(position.x, position.y)
            .build();
        let rigid_body_handle = body_set.insert(rigid_body);
        let collider_handles = colliders
            .into_iter()
            .map(|collider| collider_set.insert(collider, rigid_body_handle, body_set))
            .collect::<Vec<Index>>();
        Body {
            rigid_body_handle,
            collider_handles,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub pos: Vector3<f32>,
}

unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

pub fn load_obj(path: &str) -> Result<Vec<Vertex>, Box<dyn Error>> {
    let (models, _materials) = tobj::load_obj(path, true)?;

    for model in models.iter() {
        let vertices = model
            .mesh
            .indices
            .iter()
            .map(|i| Vertex {
                pos: nalgebra::Vector3::new(
                    model.mesh.positions[(i * 3) as usize],
                    model.mesh.positions[(i * 3 + 1) as usize],
                    model.mesh.positions[(i * 3 + 2) as usize],
                ),
            })
            .collect::<Vec<Vertex>>();
        return Ok(vertices);
    }

    Ok(vec![])
}

pub fn generate_matrix(aspect_ratio: f32) -> nalgebra::Matrix4<f32> {
    let mx_projection = nalgebra::base::Matrix::new_perspective(
        aspect_ratio,
        std::f32::consts::FRAC_PI_4,
        1.0,
        10.0,
    );
    let mx_view = nalgebra::base::Matrix::look_at_rh(
        &nalgebra::Point3::new(0f32, 0.0, 5.0),
        &nalgebra::Point3::new(0f32, 0.0, 0.0),
        &nalgebra::Vector3::<f32>::new(0.0, 1.0, 0.0),
    );
    mx_projection * mx_view
}

pub struct Renderer {
    pub device: Device,
    pub swap_chain: SwapChain,
    surface: Surface,
    sc_desc: SwapChainDescriptor,
    render_pipeline: RenderPipeline,
    bind_group: BindGroup,
    pub queue: Queue,
}

pub struct Frame {
    pub swap_chain_frame: wgpu::SwapChainFrame,
    pub command_encoder: wgpu::CommandEncoder,
}

fn create_swap_chain_descriptor(size: PhysicalSize<u32>) -> SwapChainDescriptor {
    SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Mailbox,
    }
}

pub async fn init(window: &winit::window::Window) -> Renderer {
    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let (size, surface) = unsafe {
        let size = window.inner_size();
        let surface = instance.create_surface(window);
        (size, surface)
    };

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::Default,
            compatible_surface: Some(&surface),
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                shader_validation: true,
            },
            None,
        )
        .await
        .unwrap();

    let sc_desc = create_swap_chain_descriptor(size);

    let swap_chain = device.create_swap_chain(&surface, &sc_desc);

    use std::mem;

    let vertex_size = mem::size_of::<Vertex>();

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStage::VERTEX,
            ty: wgpu::BindingType::UniformBuffer {
                dynamic: false,
                min_binding_size: wgpu::BufferSize::new(64),
            },
            count: None,
        }],
        label: None,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(""),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let mx_total = generate_matrix(sc_desc.width as f32 / sc_desc.height as f32);
    let mx_ref: &[[f32; 4]; 4] = mx_total.as_ref();
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: bytemuck::cast_slice(mx_ref),
        usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(uniform_buf.slice(..)),
        }],
        label: None,
    });

    let vs_module = device.create_shader_module(wgpu::include_spirv!(
        "../assets/gen/shaders/shader.vert.spv"
    ));

    let fs_module = device.create_shader_module(wgpu::include_spirv!(
        "../assets/gen/shaders/shader.frag.spv"
    ));

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(""),
        layout: Some(&pipeline_layout),
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &vs_module,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &fs_module,
            entry_point: "main",
        }),
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::None,
            ..Default::default()
        }),
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[wgpu::ColorStateDescriptor {
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            color_blend: wgpu::BlendDescriptor::REPLACE,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            write_mask: wgpu::ColorWrite::ALL,
        }],
        depth_stencil_state: None,
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: (&[wgpu::VertexBufferDescriptor {
                stride: vertex_size as wgpu::BufferAddress,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: (&[wgpu::VertexAttributeDescriptor {
                    format: wgpu::VertexFormat::Float3,
                    offset: 0,
                    shader_location: 0,
                }]),
            }]),
        },
        sample_count: 1,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    });

    Renderer {
        sc_desc,
        swap_chain,
        device,
        surface,
        render_pipeline,
        bind_group,
        queue,
    }
}

#[derive(Default)]
struct VertexConstructor;

impl FillVertexConstructor<Vertex> for VertexConstructor {
    fn new_vertex(&mut self, position: Point, _attributes: FillAttributes) -> Vertex {
        Vertex {
            pos: Vector3::<f32>::new(position.x, position.y, 0.0),
        }
    }
}

impl Renderer {
    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.sc_desc = create_swap_chain_descriptor(size);
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
    }

    fn create_level_vertex_buffers(
        &mut self,
        level: &Level,
    ) -> ((wgpu::Buffer, wgpu::Buffer), usize) {
        let tessellated = level.tessellate::<Vertex, VertexConstructor>();
        let vertex_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex buffer"),
                contents: bytemuck::cast_slice(&tessellated.vertices),
                usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
            });
        let index_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("index buffer"),
                contents: bytemuck::cast_slice(&tessellated.indices),
                usage: wgpu::BufferUsage::INDEX | wgpu::BufferUsage::COPY_DST,
            });
        ((vertex_buf, index_buf), tessellated.indices.len())
    }

    pub fn begin_frame(&mut self) -> Frame {
        let frame = match self.swap_chain.get_current_frame() {
            Ok(frame) => frame,
            Err(_) => {
                self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
                self.swap_chain
                    .get_current_frame()
                    .expect("Failed to acquire next swap chain texture!")
            }
        };
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        Frame {
            swap_chain_frame: frame,
            command_encoder: encoder,
        }
    }

    pub fn end_frame(&mut self, frame: Frame) {
        self.queue.submit(Some(frame.command_encoder.finish()));
    }
}

pub const WINDOW_NAME: &str = "Repro";
const ACCELERATION: f32 = 3.0;
const ANGULAR_SPEED: f32 = 5.0;

#[tokio::main]
async fn main() {
    dotenv::dotenv().expect("failed to initialize dotenv");

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_env("LOG_LEVEL"))
        .init();
    info!("Starting client");

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("test")
        .with_inner_size(LogicalSize {
            width: 800.0,
            height: 600.0,
        })
        .build(&event_loop)
        .unwrap();
    let mut renderer = init(&window).await;

    let mut pipeline = PhysicsPipeline::new();
    let gravity = Vector2::new(0.0, 0.0);
    let integration_parameters = IntegrationParameters::default();
    let mut broad_phase = BroadPhase::new();
    let mut narrow_phase = NarrowPhase::new();
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut joints = JointSet::new();
    let event_handler = Box::new(());

    let player_body = Body::new(
        BodyStatus::Dynamic,
        vec![ColliderBuilder::new(Shape::Ball(Ball::new(0.3))).build()],
        Vector3::new(-1.0, 0.0, 0.0),
        &mut bodies,
        &mut colliders,
    );
    Body::new(
        BodyStatus::Static,
        vec![crate::level::TEST_LEVEL.get_collider()],
        Vector3::new(0.0, 0.0, 0.0),
        &mut bodies,
        &mut colliders,
    );

    let mut input_up = false;
    let mut input_down = false;
    let mut input_left = false;
    let mut input_right = false;

    let model = vec![
        Vertex {
            pos: Vector3::new(-0.3, -0.3, 0.0),
        },
        Vertex {
            pos: Vector3::new(0.3, -0.3, 0.0),
        },
        Vertex {
            pos: Vector3::new(-0.3, 0.3, 0.0),
        },
        Vertex {
            pos: Vector3::new(0.3, -0.3, 0.0),
        },
        Vertex {
            pos: Vector3::new(-0.3, 0.3, 0.0),
        },
        Vertex {
            pos: Vector3::new(0.3, 0.3, 0.0),
        },
    ];
    let mut vertex_data = model.clone();

    let vertex_buf = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        });
    let (level_buf, level_len) = renderer.create_level_vertex_buffers(&crate::level::TEST_LEVEL);

    let event_loop = event_loop;
    let window = window;
    let mut last_timestamp = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        if Instant::now() - last_timestamp >= Duration::new(0, (1e9 / 60 as f64).floor() as u32) {
            last_timestamp = Instant::now();

            pipeline.step(
                &gravity,
                &integration_parameters,
                &mut broad_phase,
                &mut narrow_phase,
                &mut bodies,
                &mut colliders,
                &mut joints,
                event_handler.as_ref(),
            );

            let body = bodies.get_mut(player_body.rigid_body_handle);

            if let Some(mut body) = body {
                let rotation = body.position.rotation * Vector2::y();
                if input_up {
                    body.apply_force(ACCELERATION * rotation);
                } else if input_down {
                    body.apply_force(-ACCELERATION * rotation);
                }

                if input_left {
                    body.angvel = ANGULAR_SPEED;
                } else if input_right {
                    body.angvel = -ANGULAR_SPEED;
                } else {
                    body.angvel = 0.0;
                }

                {
                    vertex_data = model
                        .iter()
                        .map(|v| {
                            let pos = body.position.rotation * Vector2::new(v.pos.x, v.pos.y)
                                + body.position.translation.vector;
                            Vertex {
                                pos: Vector3::new(pos.x, pos.y, 0.0),
                            }
                        })
                        .collect::<Vec<Vertex>>();
                }
            }
        }

        match event {
            Event::MainEventsCleared => window.request_redraw(),
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                renderer.resize(size);
            }
            Event::RedrawRequested(_) => {
                let mut frame = renderer.begin_frame();
                {
                    let command_encoder = &mut frame.command_encoder;
                    let render_pipeline = &mut renderer.render_pipeline;
                    let bind_group = &mut renderer.bind_group;
                    let queue = &mut renderer.queue;

                    let mut rpass =
                        command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            color_attachments: (&[wgpu::RenderPassColorAttachmentDescriptor {
                                attachment: &frame.swap_chain_frame.output.view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: 1.0,
                                        g: 1.0,
                                        b: 1.0,
                                        a: 1.0,
                                    }),
                                    store: true,
                                },
                            }]),
                            depth_stencil_attachment: None,
                        });

                    rpass.set_pipeline(&render_pipeline);
                    rpass.set_bind_group(0, &bind_group, &[]);
                    rpass.set_vertex_buffer(0, vertex_buf.slice(..));
                    rpass.draw(0..vertex_data.len() as u32, 0..1);

                    {
                        let (vertices, indices) = &level_buf;
                        rpass.set_vertex_buffer(0, vertices.slice(..));
                        rpass.set_index_buffer(indices.slice(..));
                        rpass.draw_indexed(0..level_len as u32, 0, 0..1);
                    }

                    queue.write_buffer(&vertex_buf, 0, bytemuck::cast_slice(&vertex_data));
                }
                renderer.end_frame(frame);
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { input, .. },
                ..
            } => {
                let evt = input;
                if let Some(keycode) = evt.virtual_keycode {
                    match keycode {
                        VirtualKeyCode::W => input_up = evt.state == ElementState::Pressed,
                        VirtualKeyCode::A => input_left = evt.state == ElementState::Pressed,
                        VirtualKeyCode::S => input_down = evt.state == ElementState::Pressed,
                        VirtualKeyCode::D => input_right = evt.state == ElementState::Pressed,
                        _ => (),
                    };
                }
            }
            _ => {}
        }
    });
}
