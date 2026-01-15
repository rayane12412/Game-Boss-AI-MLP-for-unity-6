using Unity.FPS.Game;
using UnityEngine;
using UnityEngine.AI; // Required to move the enemy using NavMesh

namespace Unity.FPS.AI
{
    [RequireComponent(typeof(EnemyController))]
    [RequireComponent(typeof(NavMeshAgent))]
    public class EnemyTurret : MonoBehaviour
    {
        public enum AIState
        {
            Idle,
            Attack
        }

        [Header("Adaptive Behavior (Neural Intelligence)")]

        [Tooltip("Base movement speed before any adaptation")]
        public float BaseMoveSpeed = 3.5f;

        [Tooltip("Distance where the enemy feels most comfortable fighting")]
        public float IdealCombatRange = 10f;

        [Tooltip("Maximum distance used to normalize perception values")]
        public float MaxSenseDistance = 35f;

        [Tooltip("Base rotation speed when aiming at the player")]
        public float BaseAimRotationSharpness = 5f;

        [Tooltip("Smoother rotation when just tracking the target")]
        public float LookAtRotationSharpness = 2.5f;

        [Tooltip("Base delay between shots")]
        public float BaseFireDelay = 1f;

        [Tooltip("How fast the brain learns (higher = faster but less stable)")]
        [Range(0.0001f, 0.05f)]
        public float LearningRate = 0.01f;

        [Tooltip("How strong rewards and penalties influence learning")]
        [Range(0.1f, 5f)]
        public float RewardScale = 1.0f;

        [Header("Reward / Punishment Values keep it betwen 0 and 1")]

        [Tooltip("Reward: being near ideal range is good")]
        [SerializeField] private float RewardNearIdealRange = 0.03f;

        [Tooltip("Reward: tracking fast players is encouraged")]
        [SerializeField] private float RewardTrackFastPlayer = 0.01f;

        [Tooltip("Punishment: being too far is bad")]
        [SerializeField] private float PunishTooFarDistance = 0.02f;

        [Tooltip("Strong punishment: whatever we were doing was not good")]
        [SerializeField] private float PunishOnDamaged = 0.35f;

        [Tooltip("Losing the player is a small failure")]
        [SerializeField] private float PunishOnLostTarget = 0.08f;

        [Header("Turret & Aiming")]
        public Transform TurretPivot;
        public Transform TurretAimPoint;
        public Animator Animator;

        [Tooltip("Time used to smoothly reset the turret when losing the target")]
        public float AimingTransitionBlendTime = 1f;

        [Header("Visual & Audio Feedback")]
        public ParticleSystem[] RandomHitSparks;
        public ParticleSystem[] OnDetectVfx;
        public AudioClip OnDetectSfx;

        public AIState AiState { get; private set; }

        // Core components
        EnemyController m_EnemyController;
        Health m_Health;
        NavMeshAgent m_NavMeshAgent;

        // Aiming math helpers
        Quaternion m_RotationWeaponForwardToPivot;
        Quaternion m_PreviousPivotAimingRotation;
        Quaternion m_PivotAimingRotation;

        float m_TimeStartedDetection;
        float m_TimeLostDetection;

        // The neural brain that adapts behavior over time
        MLP m_Brain;

        // Used to estimate player movement speed
        Vector3 m_LastTargetPos;
        float m_LastTargetSampleTime;

        // Accumulated reward/punishment before training
        float m_RewardSignal = 0f;

        const string k_AnimOnDamagedParameter = "OnDamaged";
        const string k_AnimIsActiveParameter = "IsActive";

        void Start()
        {
            // Grab required components
            m_Health = GetComponent<Health>();
            m_EnemyController = GetComponent<EnemyController>();
            m_NavMeshAgent = GetComponent<NavMeshAgent>();

            // Safety checks
            DebugUtility.HandleErrorIfNullGetComponent<Health, EnemyTurret>(m_Health, this, gameObject);
            DebugUtility.HandleErrorIfNullGetComponent<EnemyController, EnemyTurret>(m_EnemyController, this, gameObject);
            DebugUtility.HandleErrorIfNullGetComponent<NavMeshAgent, EnemyTurret>(m_NavMeshAgent, this, gameObject);

            // Subscribe to events
            m_Health.OnDamaged += OnDamaged;
            m_EnemyController.onDetectedTarget += OnDetectedTarget;
            m_EnemyController.onLostTarget += OnLostTarget;

            // Initial NavMesh setup
            m_NavMeshAgent.speed = BaseMoveSpeed;
            m_NavMeshAgent.stoppingDistance = IdealCombatRange;
            m_NavMeshAgent.updateRotation = false; // Rotation is handled manually

            // Used to correctly align weapon rotation with turret pivot
            m_RotationWeaponForwardToPivot =
                Quaternion.Inverse(m_EnemyController.GetCurrentWeapon().WeaponMuzzle.rotation) *
                TurretPivot.rotation;

            AiState = AIState.Idle;
            m_PreviousPivotAimingRotation = TurretPivot.rotation;
            m_TimeStartedDetection = Mathf.NegativeInfinity;

            // Create a neural network:
            // Inputs: distance, player speed, target visibility, distance error
            // Outputs: movement speed, aim speed, fire rate
            m_Brain = new MLP(input: 4, hidden: 8, output: 3, seed: 1234);

            m_LastTargetSampleTime = Time.time;
        }

        void Update()
        {
            UpdateMovement();
            UpdateCombatLogic();
        }

        void LateUpdate()
        {
            UpdateTurretAiming();
        }

        void UpdateMovement()
        {
            if (AiState == AIState.Attack && m_EnemyController.KnownDetectedTarget != null)
            {
                // Chase the player while respecting NavMesh rules
                m_NavMeshAgent.isStopped = false;
                m_NavMeshAgent.SetDestination(
                    m_EnemyController.KnownDetectedTarget.transform.position
                );
            }
            else
            {
                // No target = stop moving
                m_NavMeshAgent.isStopped = true;
            }
        }

        void UpdateCombatLogic()
        {
            if (AiState != AIState.Attack || m_EnemyController.KnownDetectedTarget == null)
                return;

            Transform target = m_EnemyController.KnownDetectedTarget.transform;

            // Distance to the player (normalized)
            float distance = Vector3.Distance(TurretAimPoint.position, target.position);
            float distanceN = Mathf.Clamp01(distance / MaxSenseDistance);

            // Estimate player speed based on position delta
            float playerSpeedN = 0f;
            float dt = Time.time - m_LastTargetSampleTime;
            if (dt > 0.05f)
            {
                Vector3 delta = target.position - m_LastTargetPos;
                float speed = delta.magnitude / dt;
                playerSpeedN = Mathf.Clamp01(speed / 10f);

                m_LastTargetPos = target.position;
                m_LastTargetSampleTime = Time.time;
            }

            // How far we are from the ideal combat distance
            float distanceError =
                Mathf.Abs(distance - IdealCombatRange) / IdealCombatRange;
            float distanceErrorN = Mathf.Clamp01(distanceError);

            // Build neural network inputs
            float[] inputs =
            {
                distanceN,
                playerSpeedN,
                1f, // target is visible
                distanceErrorN
            };

            // Let the brain decide how aggressive to be
            float[] outputs = m_Brain.Forward(inputs);

            float moveMultiplier = Mathf.Lerp(0.9f, 1.8f, outputs[0]);
            float aimMultiplier = Mathf.Lerp(0.8f, 3.0f, outputs[1]);
            float fireMultiplier = Mathf.Lerp(0.6f, 2.2f, outputs[2]);

            // Apply movement adaptation
            m_NavMeshAgent.speed = BaseMoveSpeed * moveMultiplier;
            m_NavMeshAgent.acceleration = Mathf.Lerp(8f, 18f, outputs[0]);

            float aimSharpness = BaseAimRotationSharpness * aimMultiplier;
            float fireDelay = BaseFireDelay / fireMultiplier;

            bool canShoot = Time.time > m_TimeStartedDetection + fireDelay;

            // Rotate turret toward target
            Vector3 directionToTarget =
                (target.position - TurretAimPoint.position).normalized;

            Quaternion targetRotation =
                Quaternion.LookRotation(directionToTarget) *
                m_RotationWeaponForwardToPivot;

            m_PivotAimingRotation = Quaternion.Slerp(
                m_PreviousPivotAimingRotation,
                targetRotation,
                (canShoot ? aimSharpness : LookAtRotationSharpness) * Time.deltaTime
            );

            if (canShoot)
            {
                Vector3 correctedDirection =
                    (m_PivotAimingRotation *
                     Quaternion.Inverse(m_RotationWeaponForwardToPivot)) *
                    Vector3.forward;

                m_EnemyController.TryAtack(
                    TurretAimPoint.position + correctedDirection
                );

                // Reward logic:
                // - Being near ideal range is good
                // - Tracking fast players is encouraged
                // - Being too far is bad
                float reward = 0f;
                reward += RewardNearIdealRange * (1f - distanceErrorN);
                reward += RewardTrackFastPlayer * playerSpeedN;
                reward -= PunishTooFarDistance * distanceN;

                m_RewardSignal += reward * RewardScale;

                TrainBrain(inputs, outputs, m_RewardSignal);
                m_RewardSignal = 0f;

                m_TimeStartedDetection = Time.time;
            }
        }

        void TrainBrain(float[] inputs, float[] outputs, float reward)
        {
            // Convert reward into targets:
            // Positive reward reinforces the decision
            // Negative reward pushes the brain away from it
            float r = Mathf.Clamp(reward, -1f, 1f);
            float[] targets = new float[3];

            for (int i = 0; i < 3; i++)
            {
                float direction = outputs[i] - 0.5f;
                targets[i] = Mathf.Clamp01(outputs[i] + r * direction);
            }

            m_Brain.Train(inputs, targets, LearningRate);
        }

        void UpdateTurretAiming()
        {
            if (AiState == AIState.Attack)
            {
                TurretPivot.rotation = m_PivotAimingRotation;

                // Rotate the whole body slowly to face the player
                Vector3 lookPos = m_EnemyController.KnownDetectedTarget.transform.position;
                lookPos.y = transform.position.y;

                transform.rotation = Quaternion.Slerp(
                    transform.rotation,
                    Quaternion.LookRotation(lookPos - transform.position),
                    Time.deltaTime * 5f
                );
            }
            else
            {
                // Smoothly relax turret when target is lost
                TurretPivot.rotation = Quaternion.Slerp(
                    m_PivotAimingRotation,
                    TurretPivot.rotation,
                    (Time.time - m_TimeLostDetection) / AimingTransitionBlendTime
                );
            }

            m_PreviousPivotAimingRotation = TurretPivot.rotation;
        }

        void OnDamaged(float dmg, GameObject source)
        {
            // Strong punishment: whatever we were doing was not good
            m_RewardSignal -= PunishOnDamaged * RewardScale;

            if (RandomHitSparks.Length > 0)
            {
                RandomHitSparks[Random.Range(0, RandomHitSparks.Length)].Play();
            }

            Animator.SetTrigger(k_AnimOnDamagedParameter);
        }

        void OnDetectedTarget()
        {
            AiState = AIState.Attack;

            foreach (var fx in OnDetectVfx) fx.Play();
            if (OnDetectSfx)
                AudioUtility.CreateSFX(
                    OnDetectSfx,
                    transform.position,
                    AudioUtility.AudioGroups.EnemyDetection,
                    1f
                );

            Animator.SetBool(k_AnimIsActiveParameter, true);
            m_TimeStartedDetection = Time.time;

            m_LastTargetPos = m_EnemyController.KnownDetectedTarget.transform.position;
            m_LastTargetSampleTime = Time.time;
        }

        void OnLostTarget()
        {
            AiState = AIState.Idle;

            foreach (var fx in OnDetectVfx) fx.Stop();

            Animator.SetBool(k_AnimIsActiveParameter, false);
            m_TimeLostDetection = Time.time;

            // Losing the player is a small failure
            m_RewardSignal -= PunishOnLostTarget * RewardScale;
        }

        //Neural Network
        class MLP
        {
            int I, H, O;
            float[,] w1;
            float[] b1;
            float[,] w2;
            float[] b2;

            float[] lastInput;
            float[] lastHidden;
            float[] lastOutput;

            System.Random rng;

            public MLP(int input, int hidden, int output, int seed)
            {
                I = input;
                H = hidden;
                O = output;

                w1 = new float[I, H];
                b1 = new float[H];
                w2 = new float[H, O];
                b2 = new float[O];

                rng = new System.Random(seed);
                Init(w1);
                Init(w2);
                Init(b1);
                Init(b2);
            }

            void Init(float[,] m)
            {
                for (int i = 0; i < m.GetLength(0); i++)
                    for (int j = 0; j < m.GetLength(1); j++)
                        m[i, j] = (float)(rng.NextDouble() * 2 - 1) * 0.15f;
            }

            void Init(float[] v)
            {
                for (int i = 0; i < v.Length; i++)
                    v[i] = (float)(rng.NextDouble() * 2 - 1) * 0.15f;
            }

            float Sigmoid(float x) => 1f / (1f + Mathf.Exp(-x));

            public float[] Forward(float[] input)
            {
                lastInput = input;
                lastHidden = new float[H];
                lastOutput = new float[O];

                for (int j = 0; j < H; j++)
                {
                    float sum = b1[j];
                    for (int i = 0; i < I; i++)
                        sum += input[i] * w1[i, j];
                    lastHidden[j] = Sigmoid(sum);
                }

                for (int k = 0; k < O; k++)
                {
                    float sum = b2[k];
                    for (int j = 0; j < H; j++)
                        sum += lastHidden[j] * w2[j, k];
                    lastOutput[k] = Sigmoid(sum);
                }

                return lastOutput;
            }

            public void Train(float[] input, float[] target, float lr)
            {
                float[] output = Forward(input);

                float[] dOut = new float[O];
                for (int k = 0; k < O; k++)
                    dOut[k] = (output[k] - target[k]) * output[k] * (1f - output[k]);

                for (int k = 0; k < O; k++)
                {
                    b2[k] -= lr * dOut[k];
                    for (int j = 0; j < H; j++)
                        w2[j, k] -= lr * dOut[k] * lastHidden[j];
                }

                float[] dHidden = new float[H];
                for (int j = 0; j < H; j++)
                {
                    float sum = 0f;
                    for (int k = 0; k < O; k++)
                        sum += dOut[k] * w2[j, k];
                    dHidden[j] = sum * lastHidden[j] * (1f - lastHidden[j]);
                }

                for (int j = 0; j < H; j++)
                {
                    b1[j] -= lr * dHidden[j];
                    for (int i = 0; i < I; i++)
                        w1[i, j] -= lr * dHidden[j] * lastInput[i];
                }
            }
        }
    }
}
